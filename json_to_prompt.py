from typing import List, Union
from dataclasses import dataclass
import os
import json
import requests
import re

def create_prompt(json_data):
    # Your JSON object

    #desired_order = ['prompt', 'negativePrompt']
    desired_order = ['prompt', 'negativePrompt', 'steps', 'sampler', 'cfgScale', 'seed', 'Model', 'Clip skip','Size', 'clipSkip','Model hash','Version']
    
    mystr = ""
    Model_Hash = None
    VAE = None
    mystr = None
    # Include all other keys in any order afterwards
    additional_keys = [key for key in json_data if key not in desired_order]
    for key in additional_keys:
        if key == 'hashes':
              #, Model hash: 6e7d18a129
              print(json_data['hashes']['model'])
              Model_Hash = f"Model hash: {json_data['hashes']['model']}"

        if key == 'resources':
             for resource in json_data['resources']:
                    if resource['type'] == 'model':
                        print("hi1")
                        #, VAE hash: 63aeecb90f, VAE: vae-ft-mse-840000-ema-pruned.safetensors
                        VAE = f"VAE: {resource['name']}.safetensors, VAE hash: {resource['hash']}"
                    elif resource['type'] == 'lora':
                        #<lora:33081_282419_annpossible:1>
                        if mystr == None:
                            print("hi2")
                            mystr = f"<lora:{resource['name']}:{resource['weight']}>"
                        else:
                            print("hi3")
                            mystr = f"<lora:{resource['name']}:{resource['weight']}>" + "," + mystr
                             
    #formatted_string += ",".join([f"{key}: {json_data[key]}" for key in additional_keys])

    if mystr != None and mystr not in json_data['prompt']:
        json_data['prompt'] = json_data['prompt'] + ',' + mystr

    for key in ['prompt', 'negativePrompt']:
        if key in json_data:
            json_data[key] += '\n'
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

    if Model_Hash != None:
         formatted_string = formatted_string + ',' + Model_Hash
    if VAE != None:
         formatted_string = formatted_string + ',' + VAE
    if Model_Hash != None:
         formatted_string = formatted_string + ',' + Model_Hash
    # Print or use the formatted string as needed
    print(formatted_string)
    return formatted_string

def walkfoldertree():
    global search_folder

    headers = {}
    headers['Content-Type'] =  'application/json'

    for root, dirs, files in os.walk(search_folder):
        for filename in files:
            if '.json' in filename or '.civitai.info' in filename:
                fullfilepath = os.path.join(root,filename)


                #if '.civitai.info' in filename or 'safetensors.json' in filename:
                if '.civitai.info' in filename :

                    result = re.search(r'\d+_\d+_(.*)', filename)
                    if result:
                        originalfilenamebeforeanyperiods = result.group(1)

                    with open(fullfilepath, 'r', encoding='utf-8') as file:
                        json_data = json.load(file)

                    base_name, ext = os.path.splitext(originalfilenamebeforeanyperiods)
                    if 'id' in json_data and 'modelId' in json_data:
                        id = json_data['id']
                        model_id = json_data['modelId']

                        if 'images' in json_data:
                            for index,image in enumerate(json_data['images']):
                                imagename,imageext = os.path.splitext(image['url'])
                                metadata = create_prompt(image['meta'])
                                print(image['url'])

                                download_image_fullpath = os.path.join(dumpdir,f"{model_id}_{id}_{index}.{imageext}")
                                download_prompt_fullpath = os.path.join(dumpdir,f"{model_id}_{id}_{index}.txt")

                                if not os.path.exists(download_prompt_fullpath):
                                    with open(download_prompt_fullpath, 'w', newline='\r\n', encoding='utf-8') as file2:
                                                file2.write(metadata)
                                    with open(allprompts, 'a', newline='\r\n', encoding='utf-8') as file2:
                                                file2.write(f"{metadata}\r\n")

                                if not os.path.exists(download_image_fullpath):
                                    response = requests.get(image['url'], headers=headers)

                                    if response.status_code == 200:
                                        # The request was successful
                                        file_size = int(response.headers.get("content-length", 0))
                                        with open(download_image_fullpath, 'wb') as file2:
                                                    file2.write(response.content)                              
                                    else:
                                        print(f"error {response.status_code}" )
                    else:
                         print("invalid data")

label_dictionary = []
search_folder = 'X:/dif/stable-diffusion-webui-docker/data/models/Lora'
dumpdir = 'c:/users/simon/desktop/test/'
allprompts = 'c:/users/simon/desktop/test/allprompts.txt'
walkfoldertree()