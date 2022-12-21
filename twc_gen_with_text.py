import torch
from tqdm.auto import tqdm
import pdb
import argparse
import time

#this is a copy from the notebook

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
import matplotlib.pyplot as plt


def gen(params):
        prompt = params["prompt"]
        output_file = params["output"]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('creating base model...')
        base_name = 'base40M-textvec'
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

        print('creating upsample model...')
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

        print('downloading base checkpoint...')
        base_model.load_state_dict(load_checkpoint(base_name, device))

        print('downloading upsampler checkpoint...')
        upsampler_model.load_state_dict(load_checkpoint('upsample', device))

        sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 0.0],
            model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
        )



        # Produce a sample from the model.
        start = time.time()
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
            samples = x


        pc = sampler.output_to_point_clouds(samples)[0] # the step above took 1 hr 56 m on a CPU (Colab did not have a CPU available)
        fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
        plt.savefig(output_file)
        end = time.time()
        print("Compute time:",round((end -start),1)," secs")
        print(f"Output saved in {output_file} for prompt:{prompt}")



def main():
    parser = argparse.ArgumentParser(description='Generate 3D object given text input',formatter_class=argparse.                   ArgumentDefaultsHelpFormatter)
    parser.add_argument('-prompt', action="store", dest="prompt",default="a red motorcycle ",help='Specify input text')
    parser.add_argument('-output', action="store", dest="output",default="output.png",help='Output file with 3D image in different angles')
    results = parser.parse_args()
    gen(vars(results))

if __name__ == '__main__':
    main()
