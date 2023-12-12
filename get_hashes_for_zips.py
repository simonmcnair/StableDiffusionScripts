import os
from collections import defaultdict
from PIL import Image, PngImagePlugin

# initialize prompt counts dictionary
prompt_counts = defaultdict(lambda: defaultdict(int))

# loop through all PNG files in current working directory
for filename in os.listdir('.'):
    if filename.endswith('.png'):
        # open image file
        with Image.open(filename) as im:
            # extract PNG text chunks
            chunks = PngImagePlugin.PngInfo(im.info).text

            # process each text chunk
            for chunk in chunks:
                # extract parameters and negative prompt if present
                if 'parameters' in chunk:
                    params_str = chunk.split('parameters,')[1]
                    params = [p.strip() for p in params_str.split(',') if p.strip()]
                    for p in params:
                        prompt_counts['positive'][p] += 1
                if 'Negative prompt' in chunk:
                    neg_str = chunk.split('Negative prompt:')[1]
                    neg = [n.strip() for n in neg_str.split(',') if n.strip()]
                    for n in neg:
                        prompt_counts['negative'][n] += 1

                # extract other prompts
                if 'Steps:' in chunk:
                    steps_str = chunk.split('Steps:')[1]
                    steps = steps_str.split(',')[0].strip()
                    prompt_counts['extra']['steps,' + steps] += 1
                if 'Sampler:' in chunk:
                    sampler_str = chunk.split('Sampler:')[1]
                    sampler = sampler_str.split(',')[0].strip()
                    prompt_counts['extra']['Sampler,' + sampler] += 1
                if 'CFG scale:' in chunk:
                    cfg_str = chunk.split('CFG scale:')[1]
                    cfg = cfg_str.split(',')[0].strip()
                    prompt_counts['extra']['CFG scale,' + cfg] += 1
                if 'Seed:' in chunk:
                    seed_str = chunk.split('Seed:')[1]
                    seed = seed_str.split(',')[0].strip()
                    prompt_counts['extra']['Seed,' + seed] += 1
                if 'Size:' in chunk:
                    size_str = chunk.split('Size:')[1]
                    size = size_str.split(',')[0].strip()
                    prompt_counts['extra']['Size,' + size] += 1
                if 'Model hash:' in chunk:
                    hash_str = chunk.split('Model hash:')[1]
                    model_hash = hash_str.split(',')[0].strip()
                    prompt_counts['extra']['Model hash,' + model_hash] += 1
                if 'Model:' in chunk:
                    model_str = chunk.split('Model:')[1]
                    model = model_str.split(',')[0].strip()
                    prompt_counts['extra']['Model,' + model] += 1

# print cumulative prompt counts
for prompt_type, prompt_vals in prompt_counts.items():
    for prompt_val, count in sorted(prompt_vals.items(), key=lambda x: x[1], reverse=True):
        print(f'{prompt_val},{prompt_type},{count}')
