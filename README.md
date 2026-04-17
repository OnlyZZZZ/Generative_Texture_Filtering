# Generative texture filtering.
This is official code of Generative Texture Filtering (SIGGRAPH 2026 conference paper).
## Structure
```
├── input                  <- Place the input images here
├── output                  <- The output results are saved here
├── infer.py                   <- Code for inference
├── requirement.txt       <- Env file
└── README.md
```

## Installation
```
conda create -n GTF python=3.10
pip install -r requirement.txt
```

## Download Qwen-Image-Edit-2509
Please manually download Qwen-Image-Edit-2509 model weights from the official Hugging Face repository at: https://huggingface.co/Qwen/Qwen-Image-Edit-2509/tree/main or from modelscope repository at:https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2509
```
├── Qwen-Image-Edit-2509                  
├──── model_index.json
├──── vae
├──── transformer
├──── tokenizer
├──── text_encoder
├──── scheduler
├──── processor
```

## Download pre-trained lora model for texture filtering
Download the checkpoint folder from https://drive.google.com/drive/folders/1iYWIYosPxiKz0ok9wrz_ZjTlFmKsPHs0
```
├── checkpoint                  
├──── adapter_config.json
├──── adapter_model.safetensors
```

## Inference
```
python infer.py \
--pretrained_model_path "./Qwen-Image-Edit-2509" \
--model_path ./checkpoint \
--input_dir ./input \
--output_dir ./output \
--num_inference_steps 4 
```
