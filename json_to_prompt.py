from typing import List, Union
from dataclasses import dataclass
import os
import json
import requests
import re
import time
from time import sleep
from PIL import Image

def replace_width_with_bob(url):
    # Use regular expression to replace /width=* with /bob/
    modified_url = re.sub(r'/width=\d+', '/original=true', url)

    return modified_url

def download_with_retry(url, max_retries=3, retry_delay=5):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Check HTTP status code
            if response.status_code == 200:
                # Process the content here
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        # Process each chunk as needed
                        print(chunk)

                # Check for unexpected connection termination
                if not response.content:
                    print("Connection closed unexpectedly")

                return True  # Download successful

        except requests.exceptions.RequestException as e:
            print(f"Error during attempt {attempt}: {e}")

        # Retry after a delay
        if attempt < max_retries:
            print(f"Retrying in {retry_delay} seconds...")
            sleep(retry_delay)

    print("Max retries reached. Download failed.")
    return False  # Download unsuccessful

# Example usage
#success = download_with_retry(url)

#if success:
#    print("Download successful!")
#else:
#    print("Download failed after retries.")

def create_prompt(json_data):
    # Your JSON object

    if len(json_data) == 0:
        return None
    #desired_order = ['prompt', 'negativePrompt']
    desired_order = ['prompt', 'negativePrompt', 'steps', 'sampler', 'CFG scale', 'cfgScale','seed', 'Model', 'Clip skip','Size', 'clipSkip','Model hash','Version']
    
    mystr = ""
    Model_Hash = None
    VAE = None
    mystr = None
    # Include all other keys in any order afterwards
    additional_keys = [key for key in json_data if key not in desired_order]
    for key in additional_keys:
        if key == 'hashes':
              #, Model hash: 6e7d18a129
              if 'embed' in json_data:
                embed_keys = [key.split('embed:', 1)[1] for key in json_data['hashes'] if key.startswith('embed:')]
                print("test")

              if 'model' in json_data:
                print(json_data['hashes']['model'])
                Model_Hash = f"Model hash: {json_data['hashes']['model']}"
        else:
             print(f"Additional Key: {key}")
        if key == 'resources':
             for resource in json_data['resources']:
                    if resource['type'] == 'model':
                        #print("hi1")
                        #, VAE hash: 63aeecb90f, VAE: vae-ft-mse-840000-ema-pruned.safetensors
                        VAE = f"VAE: {resource['name']}.safetensors, VAE hash: {resource['hash']}"
                    elif resource['type'] == 'lora' and 'name' in resource and 'weight' in resource:
                        #<lora:33081_282419_annpossible:1>
                        if mystr == None:
                            #print("hi2")
                            mystr = f"<lora:{resource['name']}:{resource['weight']}>"
                        else:
                            #print("hi3")
                            mystr = f"<lora:{resource['name']}:{resource['weight']}>" + "," + mystr
                            
    #formatted_string += ",".join([f"{key}: {json_data[key]}" for key in additional_keys])
    try:
        if 'prompt' in json_data:
            if mystr != None and mystr not in json_data['prompt']:
                json_data['prompt'] = json_data['prompt'] + ',' + mystr

            for key in ['prompt', 'negativePrompt']:
                if key in json_data:
                    json_data[key] += '\n'
        else:
            print("No prompt")
            #return None
    except Exception as e:
         print(f"error {e}")
         return False
    # Create a formatted string with "prompt" and "negativePrompt" first
    formatted_string = ",".join([f"{key}: {json_data[key]}" for key in desired_order if key in json_data])

    formatted_string = formatted_string.replace('  ',' ')
    formatted_string = formatted_string.replace(', ',',')
    formatted_string = formatted_string.replace(' ,',',')
    formatted_string = formatted_string.replace('prompt: ','')
    formatted_string = formatted_string.replace(',negativePrompt: ','Negative prompt: ' )
    formatted_string = formatted_string.replace(',clipSkip: ',',Clip skip: ' )
    formatted_string = formatted_string.replace(',ClipSkip: ',',Clip skip: ' )
    formatted_string = formatted_string.replace(',clipskip: ',',Clip skip: ' )
    formatted_string = formatted_string.replace(',cfgScale: ',',CFG scale: ' )

    if Model_Hash != None:
         formatted_string = formatted_string + ',' + Model_Hash
    if VAE != None:
         formatted_string = formatted_string + ',' + VAE
    if Model_Hash != None:
         formatted_string = formatted_string + ',' + Model_Hash
    # Print or use the formatted string as needed
    print(formatted_string)
    if 'prompt' not in json_data and 'Seed' not in json_data and 'Sampler' not in json_data:
        return None
    else:
        return formatted_string

def unquote(text):
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text

    try:
        return json.loads(text)
    except Exception:
        return text
    
def parse_generation_parameters(x: str):
    """parses generation parameters string, the one you see in text field under the picture in UI:
```
girl with an artist's beret, determined, blue eyes, desert scene, computer monitors, heavy makeup, by Alphonse Mucha and Charlie Bowater, ((eyeshadow)), (coquettish), detailed, intricate
Negative prompt: ugly, fat, obese, chubby, (((deformed))), [blurry], bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), messy drawing
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 965400086, Size: 512x512, Model hash: 45dee52b
```

    returns a dict with field values
    """

    re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
    re_param = re.compile(re_param_code)
    re_imagesize = re.compile(r"^(\d+)x(\d+)$")
    re_hypernet_hash = re.compile("\(([0-9a-f]+)\)$")
    if 'Template' in x:
        print("has template")

    res = {}

    prompt = ""
    negative_prompt = ""

    done_with_prompt = False

    *lines, lastline = x.strip().split("\n")
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ''

    for line in lines:
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()
        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    res["Prompt"] = prompt
    res["Negative prompt"] = negative_prompt

    for k, v in re_param.findall(lastline):
        try:
            if len(v) > 0:
                if v[0] == '"' and v[-1] == '"':
                    v = unquote(v)

                m = re_imagesize.match(v)
                if m is not None:
                    res[f"{k}-1"] = m.group(1)
                    res[f"{k}-2"] = m.group(2)
                else:
                    res[k] = v
            else:
                print(f"ignoring key {k} as value is empty")
        except Exception as e:
            print(f"Error parsing \"{k}: {v}\" {e}")

    # Missing CLIP skip means it was set to 1 (the default)
    if "Clip skip" not in res:
        res["Clip skip"] = "1"

    hypernet = res.get("Hypernet", None)
    if hypernet is not None:
        res["Prompt"] += f"""<hypernet:{hypernet}:{res.get("Hypernet strength", "1.0")}>"""

    if "Hires resize-1" not in res:
        res["Hires resize-1"] = 0
        res["Hires resize-2"] = 0

    if "Hires sampler" not in res:
        res["Hires sampler"] = "Use same sampler"

    if "Hires checkpoint" not in res:
        res["Hires checkpoint"] = "Use same checkpoint"

    if "Hires prompt" not in res:
        res["Hires prompt"] = ""

    if "Hires negative prompt" not in res:
        res["Hires negative prompt"] = ""

    #restore_old_hires_fix_params(res)

    # Missing RNG means the default was set, which is GPU RNG
    if "RNG" not in res:
        res["RNG"] = "GPU"

    if "Schedule type" not in res:
        res["Schedule type"] = "Automatic"

    if "Schedule max sigma" not in res:
        res["Schedule max sigma"] = 0

    if "Schedule min sigma" not in res:
        res["Schedule min sigma"] = 0

    if "Schedule rho" not in res:
        res["Schedule rho"] = 0

    if "VAE Encoder" not in res:
        res["VAE Encoder"] = "Full"

    if "VAE Decoder" not in res:
        res["VAE Decoder"] = "Full"

    #skip = set(shared.opts.infotext_skip_pasting)
    #res = {k: v for k, v in res.items() if k not in skip}

    return res

def write_if_not_exists(file_path, text_to_write):
    # Read the contents of the file
    if os.path.exists(file_path):
        with open(file_path, 'r', newline='\r\n', encoding='utf-8') as file:
            existing_content = file.read()

            # Check if the text_to_write already exists in the file
            if text_to_write not in existing_content:
                print("Append the text to the file")
                with open(file_path, 'a', newline='\r\n', encoding='utf-8') as file:
                    file.write(text_to_write + '\r\n')  # Adding a newline for separation
            else:
                print("Prompt already exists in file")
    else:
        print("File doesn't exist.  Create")
        with open(file_path, 'a', newline='\r\n', encoding='utf-8') as file:
            file.write(text_to_write + '\r\n')  # Adding a newline for separation

def has_parameters(filepath, extended=False):

    if os.path.exists(filepath) and os.path.isfile(filepath):
        if filepath.endswith(".png"):
            with Image.open(filepath) as img:
                try:
                    parameter = img.info.get("parameters")
                    if parameter is not None:
                        print(filepath + " has metadata.")
                        if extended == True:
                            res = parse_generation_parameters(parameter)
                            if 'Prompt' in res and 'Seed' in res and 'Sampler' in res:
                                #highlevel looks valid
                                return res, True
                            else:
                                print("insufficient parameters to be considered a prompt")
                                return None, False
                        else:
                            return True
                    else:
                        print("PNG with no metadata")
                        if extended == True:
                            return None,False
                        else:
                            return False
                except Exception as e:
                    print("damaged png file")
        else:
            print("non png files don't have parameters")
            if extended == True:
                return None, False
            else:
                return False


def get_images_for_models(treetoproc):

    headers = {}
    headers['Content-Type'] =  'application/json'
    force_image_dl = True
    png = True

    for root, dirs, files in os.walk(treetoproc):
        for filename in files:
            if  filename.endswith('.civitai.info'):
                fullfilepath = os.path.normpath(os.path.join(root,filename))

                if 'edgChicLingerie_R' in filename:
                    print("test")
                result = re.search(r'\d+_\d+_(.*)', filename)
                if result:
                    originalfilenamebeforeanyperiods = result.group(1)

                with open(fullfilepath, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)

                base_name, ext = os.path.splitext(originalfilenamebeforeanyperiods)
                base_name, ext = os.path.splitext(base_name)
                print(f"Processing {base_name}")
                if 'id' in json_data and 'modelId' in json_data:
                    id = json_data['id']
                    model_id = json_data['modelId']
                    allprompts = os.path.normpath((os.path.join(root,(f"{model_id}_{id}_{base_name}_prompts.txt"))))

                    if 'images' in json_data:
                        for index,image in enumerate(json_data['images']):
                            imagename,imageext = os.path.splitext(image['url'])
                            if 'meta' in image and image['meta'] != None and len(image['meta']) >0:
                                #we only want images or prompts with metadata

                                if image.get('type') == 'image':
                                    if 'prompt' in image['meta']:
                                        if png == True:
                                            image['url'] = image['url'].replace('.jpeg','.png')
                                            image['url'] = image['url'].replace('.jpg','.png')
                                            image['url'] = replace_width_with_bob(image['url'])
                                    else:print("No prompt.")

                                    result = re.search(r'.*\/(.*..*$)', image['url'])
                                            
                                    if result:
                                        image_file_name = result.group(1)

                                        image_file_name = image_file_name.replace('.jpeg','.png')
                                        image_file_name = image_file_name.replace('.jpg','.png')
                                        image_file_name = replace_width_with_bob(image_file_name)

                                        download_image_fullpath = os.path.normpath(os.path.join(root,f"{model_id}_{id}_{base_name}_{image_file_name}"))
                                    else:
                                        print("could not get filename from URL.")
                                        download_image_fullpath = os.path.normpath(os.path.join(root,f"{model_id}_{id}_{index}{imageext}"))

                                    if os.path.exists(download_image_fullpath) and os.path.getsize(download_image_fullpath) >0:

                                        parameter, result = has_parameters(download_image_fullpath, True)
                                        if result:
                                            print(f"image file {download_image_fullpath} already exists and has parameters")
                                        else:
                                            print(f"image file {download_image_fullpath} already exists but has no parameters, get prompt")
                                            if len(image['meta']) == 0:
                                                print("empty metadata")
                                            else:
                                                promptpath = download_image_fullpath.replace('.png','.txt').replace('.jpg','.txt').replace('.jpeg','.txt')
                                                if os.path.exists(promptpath):
                                                    print("prompt file already exists.")
                                                else:
                                                    metadata = create_prompt(image['meta'])
                                                    if metadata != None:
                                                        write_if_not_exists(promptpath,metadata)
                                                    else:
                                                        print("invalid metadata.  Removing image")
                                                        os.remove(download_image_fullpath)
                                    else:
                                        response = requests.get(image['url'], headers=headers)
                                        if response.status_code == 200:
                                            # The request was successful
                                            if response.headers['Content-Type'] == 'image/jpeg':
                                                print("jpeg")
                                                download_image_fullpath= download_image_fullpath.replace('.png','.jpg')
                                                with open(download_image_fullpath, 'wb') as file2:
                                                            file2.write(response.content)
                                                print("It's a JPG so get the metadata")
                                                promptpath = download_image_fullpath.replace('.png','.txt').replace('.jpg','.txt').replace('.jpeg','.txt')
                                                if os.path.exists(promptpath):
                                                    print("Metadata already exists")
                                                else:
                                                    metadata = create_prompt(image['meta'])
                                                    if metadata != None:
                                                        write_if_not_exists(promptpath,metadata)
                                                    else:
                                                        print("invalid metadata.  Removing image")
                                                        os.remove(download_image_fullpath)


                                                #print(response.headers['Content-Type'])
                                            elif response.headers['Content-Type'] == 'image/png':
                                                print("legit PNG")

                                                file_size = int(response.headers.get("content-length", 0))
                                                with open(download_image_fullpath, 'wb') as file2:
                                                            file2.write(response.content)

                                                parameter, result = has_parameters(download_image_fullpath, True)
                                                if result:
                                                    print("has parameters")
                                                else:
                                                    print("no parameters, use existing prompt")
                                                    promptpath = download_image_fullpath.replace('png','.txt').replace('.jpg','.txt').replace('.jpeg','.txt')
                                                    if os.path.exists(promptpath):
                                                        print("Metadata file already exists")                                                    
                                                    else:
                                                        metadata = create_prompt(image['meta'])
                                                        if metadata != None:
                                                            write_if_not_exists(promptpath,metadata)
                                                        else:
                                                            print("invalid metadata.  Removing image")
                                                            os.remove(download_image_fullpath)
                                        else:print(f"{base_name}.  error {image['url']}.  {response.status_code}" )
                                else:
                                    print(f"this isn't an image.  It's {image.get('type')}")
                            else:
                                print("this has no metadata.  Ignoring")
                    else:
                        print(f"no images for {base_name}.  error {image['url']}")

label_dictionary = []
#allprompts = 'c:/users/simon/desktop/test/allprompts.txt'
get_images_for_models('X:/dif/stable-diffusion-webui-docker/data/models/Lora')