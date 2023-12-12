import os
from PIL import Image
import csv

def extract_metadata(png_path):
    try:
        with Image.open(png_path) as im:
            metadata = im.info
            if 'parameters' in metadata:
                params = metadata['parameters']
                positive_prompts = [x.strip() for x in params.split(',') if ':' in x]
                negative_prompts = []
                if 'Negative prompt:' in params:
                    negative_prompts = [x.strip() for x in params.split('Negative prompt:')[1].split(',') if ':' in x]

                steps = metadata.get('Steps', '')
                sampler = metadata.get('Sampler', '')
                cfg_scale = metadata.get('CFG scale', '')
                seed = metadata.get('Seed', '')
                size = metadata.get('Size', '')
                model_hash = metadata.get('Model hash', '')
                model = metadata.get('Model', '')

                data = []
                for prompt in positive_prompts:
                    data.append((prompt, 'positive', 1))
                for prompt in negative_prompts:
                    data.append((prompt, 'negative', 1))

                data.append(('steps', steps, 1))
                data.append(('Sampler', sampler, 1))
                data.append(('CFG scale', cfg_scale, 1))
                data.append(('Seed', seed, 1))
                data.append(('Size', size, 1))
                data.append(('Model hash', model_hash, 1))
                data.append(('Model', model, 1))

                return data
    except Exception as e:
        print(f"Error processing {png_path}: {e}")
    return []


if __name__ == '__main__':
    png_files = [f for f in os.listdir('.') if f.endswith('.png')]

    prompt_data = []
    for png_file in png_files:
        data = extract_metadata(png_file)
        prompt_data.extend(data)

    prompt_data = sorted(prompt_data, key=lambda x: x[2], reverse=True)

    with open('prompt.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['prompt', 'type', 'count'])
        for prompt, prompt_type, count in prompt_data:
            writer.writerow([prompt, prompt_type, count])
