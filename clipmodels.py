import open_clip
#pip install open-clip-torch
import clip_interrogator
from clip_interrogator import Config, Interrogator
import torch
from PIL import Image
import os
from datasets import load_dataset
#pip install datasets
#pip install transformers
from huggingface_hub import ModelCard
import csv
import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Optional

from codetiming import Timer



def get_model_size(model):
    # Load the model information
    #dataset = load_dataset("hf://models", model)
    dataset = load_dataset(model)

    # Get the size of the model file in bytes
    model_file_size_bytes = dataset['train']['file_size']

    # Convert to gigabytes
    model_file_size_gb = model_file_size_bytes / (1024 ** 3)

    model_size = model_file_size_gb

    print(f"Model File Size: {model_file_size_gb:.2f} GB")

    return model_size

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            import gc
            #print("mem before")
            #print(torch.cuda.memory_summary(device=None, abbreviated=False))
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            #del variables

            #print("mem after")
            #print(torch.cuda.memory_summary(device=None, abbreviated=False))

def low_vram():
    if torch.cuda.is_available():
        vram_total_mb = torch.cuda.get_device_properties('cuda').total_memory / (1024**2)
        vram_info = f"GPU VRAM: **{vram_total_mb:.2f}MB**"
        if vram_total_mb< 8:
            vram_info += "<br>Using low VRAM configuration"
    if vram_total_mb <= '4': return False 
    return True

def return_vram():
    if torch.cuda.is_available():
        vram_total_mb = torch.cuda.get_device_properties('cuda').total_memory / (1024**2)
    return vram_total_mb


def load(clip_model_name):
    global ci
    res = None
    if ci is None:
        print(f"Loading CLIP Interrogator {clip_interrogator.__version__}...")

        config = Config(
            cache_path = 'models/clip-interrogator',
            clip_model_name=clip_model_name,
        )

        if low_vram:
            print("low vram")
            config.apply_low_vram_defaults()
            config.chunk_size = 512
        ci = Interrogator(config)

    if clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        with Timer() as modelloadtime:
            ci.load_clip_model()
        print(f"loading model took {modelloadtime.last} to load")
        return modelloadtime.last

    return res

def test_output(imagepath,caption=None):
    res = ['/'.join(x) for x in open_clip.list_pretrained()]

    total = len(res)
    cnt = 0
    base_folder = os.path.dirname(imagepath)
    basefilename, ext = os.path.splitext(imagepath)
    vram = return_vram
    filename = os.path.join(base_folder, 'batch.txt')
    print(f"writing to filepath {filename}")

    #res = get_model_card_size(mod)
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            second_column_values = []
            csv_reader = csv.reader(f)
            #next(csv_reader, None)
            for row in csv_reader:
                if len(row) >= 2:  # Check if the row has at least two columns
                    if 'out of memory' not in row[2]:
                        second_column_values.append(row[1]) 
                        print(f"{row[1]} Processed already.")    
                    else:
                        print(f"ignore this row {row[1]} it is an out of memory fail.  {row[2]}")
    else:
        second_column_values = []
        with open(filename, 'a', encoding='utf-8') as f:
                f.write(f"filename_Processed,modelname,Quality,model_loadtime,Inference_time,Output\r\n")
 
    for  mod in res:
        cnt += 1

        print(f"Processing model {cnt}/{total}")
        if low_vram:
            torch_gc()
            torch.cuda.empty_cache()
        print(mod)
        try:
            if mod in second_column_values:
                print(f"Already processed {mod}, skipping.")
                continue
            #mod_size = get_model_size(mod)

            #if mod_size < vram:
            with Timer() as modelloadtime:
                load(mod)
            print(f"{mod} took {modelloadtime.last} to load")
            image = Image.open(imagepath).convert('RGB')

            #image = image.convert('RGB')
            #'best':
            print("Processing #1. best")
            with Timer() as besttime:
                best = ci.interrogate(image, caption=caption)
            print(f"{mod},best,{modelloadtime.last},{besttime.last},{best}")
            # 'caption':
            print("Processing #2. caption")
            with Timer() as captiontime:
                caption = ci.generate_caption(image) if caption is None else caption
            print(f"{mod},caption,{modelloadtime.last},{captiontime.last},{caption}")
            # 'classic':
            print("Processing #3. classic")
            with Timer() as classictime:
                classic = ci.interrogate_classic(image, caption=caption)
            print(f"{mod},classic,{modelloadtime.last},{classictime.last},{classic}")
            # 'fast':
            print("Processing #4. fast")
            with Timer() as fasttime:
                fast = ci.interrogate_fast(image, caption=caption)
            print(f"{mod},fast,{modelloadtime.last},{fasttime.last},{fast}")
            # 'negative':
            print("Processing #5. negative")
            with Timer() as negativetime:
                negative = ci.interrogate_negative(image)
            print(f"{mod},negative,{modelloadtime.last},{negativetime.last},{negative}")
            print(f"writing to filepath {filename}")
            with open(filename, 'a', encoding='utf-8') as f:
                    f.write(f"{basefilename},{mod},best,{modelloadtime.last},{besttime.last},{best}\r\n")
                    f.write(f"{basefilename},{mod},caption,{modelloadtime.last},{captiontime.last},{caption}\r\n")
                    f.write(f"{basefilename},{mod},classic,{modelloadtime.last},{classictime.last},{classic}\r\n")
                    f.write(f"{basefilename},{mod},fast,{modelloadtime.last},{fasttime.last},{fast}\r\n")
                    f.write(f"{basefilename},{mod},negative,{modelloadtime.last},{negativetime.last},{negative}\r\n")#
#            else:
#                print(f"model size is {mod_size} but we only have {vram}")
        except Exception as e:
            print(f"Oops {e}")
            error = (str(e).replace('\n', '').replace('\r', ''))
            with open(filename, 'a', encoding='utf-8') as f:
                    f.write(f"{basefilename},{mod},FAILED {error}\r\n")


ci = None
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024,expandable_segments:True"

#test_output('X:\\dif\\stable-diffusion-webui-docker\\output\\txt2img\\2024-02-16\\Newfolder\\00005-ponyDiffusionV6XL_v6StartWithThisOne_30_7_15650787_None.png')
test_output('/srv/dev-disk-by-uuid-e83913b3-e590-4dc8-9b63-ce0bdbe56ee9/Stable/MonaLisaResize.jpg')


