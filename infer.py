import argparse
import math
import os
from PIL import Image

from diffusers import (
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
    QwenImageEditPlusPipeline,
)
from diffusers.models import QwenImageTransformer2DModel
import torch
import argparse
import os
from vanillaPipeline import VanillaPipeline
from peft import PeftModel
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTF inference script")
    parser.add_argument("--pretrained_model_path", type=str, required=True, help="Path to Qwen-image-2509 model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to LoRA model")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images")
    parser.add_argument("--num_inference_steps", type=int, default=8, help="Diffusion steps")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (e.g. 'cuda:0' or 'cpu')")
    args = parser.parse_args()

    torch_dtype = torch.bfloat16
    model_name = args.pretrained_model_path
    lora_weight = args.model_path
    input_dir = args.input_dir
    output_dir = args.output_dir
    device = args.device
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),  
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,  # set shift_terminal to None
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    pipe = VanillaPipeline.from_pretrained(
        model_name, scheduler=scheduler, torch_dtype=torch_dtype
    ).to(device)

    transformer = pipe.transformer
    def _unwrap(m):
        return m._orig_mod if hasattr(m, "_orig_mod") else m
    _unwrap_flux = _unwrap(transformer)
    transformer = PeftModel.from_pretrained(_unwrap_flux, lora_weight, low_cpu_mem_usage=False)

    transformer.eval()
    pipe.transformer = transformer

    input_args = {
        "prompt": "Remove the texture and preserve the structure.",
        "generator": torch.Generator(device=device).manual_seed(2026),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": args.num_inference_steps,
    }
    os.makedirs(output_dir, exist_ok=True)
    valid_ext = [".png",".jpg", ".jpeg"]
    image_files = [
        f for f in os.listdir(input_dir)
        if any(f.lower().endswith(ext) for ext in valid_ext)
    ]

    for filename in tqdm(image_files, desc="Processing images", ncols=100):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        input_args["image"] = Image.open(input_path).convert("RGB")
        image = pipe(**input_args).images[0]
        image.save(output_path)
    