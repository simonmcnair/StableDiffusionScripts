from typing import List, Union
from dataclasses import dataclass
import os
import json
import requests
import re
import time
from time import sleep

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
        if mystr != None and mystr not in json_data['prompt']:
            json_data['prompt'] = json_data['prompt'] + ',' + mystr

        for key in ['prompt', 'negativePrompt']:
            if key in json_data:
                json_data[key] += '\n'
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
    return formatted_string

def walkfoldertree(treetoproc):

    headers = {}
    headers['Content-Type'] =  'application/json'
    force_image_dl = True
    png = True

    for root, dirs, files in os.walk(treetoproc):
        for filename in files:
            if  filename.endswith('.civitai.info'):
                fullfilepath = os.path.normpath(os.path.join(root,filename))

                if 'l4t3xv3lm4' in filename:
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
                    allprompts = os.path.normpath((os.path.join(root,(f"{base_name}_prompts.txt"))))

                    if 'images' in json_data:
                        for index,image in enumerate(json_data['images']):
                            imagename,imageext = os.path.splitext(image['url'])
                            if 'meta' in image:
                                #we only want images or prompts with metadata
                                if image.get('type') == 'image':
                                    if get_prompt == True: 
                                        if image['meta'] != None:
                                            if 'prompt' in image['meta']:
                                                if png == True:
                                                    image['url'] = image['url'].replace('.jpeg','.png')
                                                    image['url'] = image['url'].replace('.jpg','.png')
                                                    image['url'] = replace_width_with_bob(image['url'])
                                                    if os.path.exists(allprompts) and os.path.getsize(allprompts) > 0:
                                                        print("prompt already exists and isn't zero")
                                                    else:
                                                        metadata = create_prompt(image['meta'])
                                                        with open(allprompts, 'w', newline='\r\n', encoding='utf-8') as file2:
                                                            file2.write(f"{metadata}\r\n")
                                        else:print("No prompt.")
                                    else:
                                        print("We're not downloading prompts")

                                    if get_images == True:

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
                                            print(f"image file {download_image_fullpath} already exists")  
                                        else:
                                            response = requests.get(image['url'], headers=headers)
                                            if response.status_code == 200:
                                                # The request was successful
                                                if response.headers['Content-Type'] == 'image/jpeg':
                                                    print("jpeg")
                                                    image['url'] = image['url'].replace('.png','.jpeg')
                                                    #print(response.headers['Content-Type'])
                                                elif response.headers['Content-Type'] == 'image/png':
                                                    print("legit PNG")

                                                file_size = int(response.headers.get("content-length", 0))
                                                with open(download_image_fullpath, 'wb') as file2:
                                                            file2.write(response.content)

                                            else:print(f"{base_name}.  error {image['url']}.  {response.status_code}" )
                                    else:
                                        print("We're not downloading images")
                                else:
                                    print(f"this isn't an image.  It's {image.get('type')}")
                            else:
                                print("this has no metadata.  Ignoring")
                    else:
                        print(f"no images for {base_name}.  error {image['url']}")

get_prompt = True
get_images = True
label_dictionary = []
#allprompts = 'c:/users/simon/desktop/test/allprompts.txt'
walkfoldertree('X:/dif/stable-diffusion-webui-docker/data/models/Lora')