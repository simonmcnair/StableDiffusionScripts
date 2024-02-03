import requests
import time
import os
import json
import re
from tqdm import tqdm
from itertools import product
import platform

from pathlib import Path

def get_operating_system():
    system = platform.system()
    return system

def sanitise_folder_name(folder_name):
    # Define a regular expression pattern to match invalid characters
    invalid_chars_pattern = re.compile(r'[\\/:"*?<>|]')

    # Replace invalid characters with an empty string
    sanitised_folder_name = re.sub(invalid_chars_pattern, '', folder_name)

    return sanitised_folder_name

def sanitise_filepath(filepath,replacewith=''):
    # Define the set of invalid characters in Windows and Linux file paths
    filepath = filepath.replace('\\','/')
    invalid_characters = set(['<', '>', '"', '\\','|', '?', '*',' '])

    # Replace or remove invalid characters
    sanitised_filepath = ''.join(char if char not in invalid_characters else replacewith for char in filepath)

    return sanitised_filepath

def get_script_name():
    # Use os.path.basename to get the base name (script name) from the full path
    #basename = os.path.basename(path)
    return Path(__file__).stem
    #return os.path.basename(__file__)

def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))

def write_to_log(log_file, message):
    print(message)
    try:
        with open(log_file, 'a', encoding='utf-8') as file:
            file.write(message + '\n')
    except Exception as e:
        print(f"Error writing to the log file: {e}")

def dump_to_json(data, filename):
    """
    Dump a Python object to a JSON file.

    Parameters:
    - data: Python object to be dumped to JSON.
    - filename: Name of the JSON file to be created.
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)  # indent for pretty formatting (optional)

def rename_file_grouping(filepath, new_filename):
    # Get the directory and base filename

    directory, old_filename = os.path.split(filepath)
    old_filename_base, old_filename_ext = os.path.splitext(filepath)
    #originalfilenamebeforeanyperiods = os.path.splitext(os.path.basename(filepath))[0]
    originalfilenamebeforeanyperiods = os.path.basename(filepath).split('.', 1)[0]

    for root, dirs, files in os.walk(directory):
        for filename in files:
            #print("processing " + filename)

            if new_filename in filename:
                print(new_filename + " already exists in " + filename + ".  Skipping")
                continue
            oldfullpath = os.path.join(root,filename)
            # Split the old filename into base and extension
            properfilename = os.path.splitext(os.path.basename(filename))[0]
            splitbyperiod = os.path.basename(filename).split('.', 1)
            filenamebeforeanyperiods = splitbyperiod[0]
            remainder = splitbyperiod[1]

            result = re.search(r'\d+_\d+_\d+_\d+_(.*)', filenamebeforeanyperiods)
            if result:
                originalfilenamebeforeanyperiods = result.group(1)
                #filenamebeforeanyperiods = os.path.normpath(os.path.join(root, desired_part))
                
            result = re.search(r'\d+_\d+_(.*)', originalfilenamebeforeanyperiods)
            if result:
                originalfilenamebeforeanyperiods = result.group(1)
                #filenamebeforeanyperiods = os.path.normpath(os.path.join(root, desired_part))

            if originalfilenamebeforeanyperiods in filenamebeforeanyperiods:
                print("filename matches prefix " + filenamebeforeanyperiods)
                # Create the new file path by joining the directory and new base filename with the original extension
                new_filepath = os.path.normpath(os.path.join(root, new_filename + "_" + originalfilenamebeforeanyperiods + "." + remainder))

                try:
                    # Rename the file
                    if oldfullpath == new_filepath:
                        print("nothing to do")
                        break
                    if not os.path.exists(new_filepath):
                        #os.rename(oldfullpath, new_filepath)
                        print(f"File successfully renamed from '{oldfullpath}' to '{new_filepath}'")
                    else:
                        print("filename already exists and will not be over written")
                except FileNotFoundError:
                    print(f"Error: File '{old_filename}' not found.")
                except Exception as e:
                    print(f"An error occurred: {e}")
            #else:
            #    print("filename does not match prefix " + filename)


def rename_files():
    global search_folder

    for root, dirs, files in os.walk(search_folder):
        for filename in files:
            if '.json' in filename or '.civitai.info' in filename:
                fullfilepath = os.path.join(root,filename)
                with open(fullfilepath, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)

                if '.civitai.info' in filename or 'safetensors.json' in filename:
                    id = json_data["id"]
                    model_id = json_data["modelId"]
                    num_models = len(json_data['files'])
                    json_filename = json_data['files'][0]['name']
                    downloadJSON_filename = f"{id}_{model_id}_{filename}.json"
                    if str(id) not in filename:
                        print("missing id")
                    if str(model_id) not in filename:
                        print("missing model_id")
                    if f"{model_id}_{id}" not in filename:

                        result = re.search(r'\d+_\d+_\d+_\d+_(.*)', filename)
                        if result:
                            originalfilenamebeforeanyperiods = result.group(1)
                            #filenamebeforeanyperiods = os.path.normpath(os.path.join(root, desired_part))
                            
                        result = re.search(r'\d+_\d+_(.*)', filename)
                        if result:
                            originalfilenamebeforeanyperiods = result.group(1)
                            newfilename = f"{id}_{model_id}_{originalfilenamebeforeanyperiods}"
                            newfilename = os.path.join(root,newfilename)
                            #filenamebeforeanyperiods = os.path.normpath(os.path.join(root, desired_part))
                        print(f"rename {fullfilepath} to {newfilename}")
                        #os.rename(fullfilepath,newfilename)
                        rename_file_grouping(fullfilepath, f"{model_id}_{id}")


                    print(downloadJSON_filename)

                elif '.json' in filename:
                    description = json_data['description']
                    if 'activation text' in json_data:
                        activation_text = json_data['activation text']
                    downloadJSON_filename = f"{description}_{activation_text}"
                    #print(downloadJSON_filename)
                                    




search_folder = '/folder/to/download/to'

current_os = get_operating_system()

if current_os == "Windows":
    print("Running on Windows")
elif current_os == "Linux":
    print("Running on Linux")

localoverridesfile = os.path.join(get_script_path(), "localoverridesfile_" + get_script_name() + '_' + current_os + '.py')

if os.path.exists(localoverridesfile):
    exec(open(localoverridesfile).read())
    #apikey = apikey
    #print("API Key:", apikey)
    print("local override file is " + localoverridesfile)

else:
    print("local override file would be " + localoverridesfile)



logfile_path = os.path.join(search_folder,'logfile.log')

successfile_path = os.path.join(search_folder,'successfile.log')

rename_files()
