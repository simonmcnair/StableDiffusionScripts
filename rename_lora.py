
from pathlib import Path
import json
import os
import re
import shutil
import time
import hashlib
import io
from blake3 import blake3
import platform

def get_operating_system():
    system = platform.system()
    return system

def get_script_name():
    # Use os.path.basename to get the base name (script name) from the full path
    #basename = os.path.basename(path)
    return Path(__file__).stem
    #return os.path.basename(__file__)

def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))

def write_to_log(log_file, message):
    global debug
    if debug == True: print(message)
    try:
        with open(log_file, 'a', encoding='utf-8') as file:
            file.write(message + '\n')
    except Exception as e:
        print(f"Error writing to the log file: {e}.  Press a key to continue")
        input()

def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()

def sha256(filename, title, use_addnet_hash=False):
    print(f"Calculating sha256 for {filename}: ", end='')
    if use_addnet_hash:
        with open(filename, "rb") as file:
            sha256_value = addnet_hash_safetensors(file)
    else:
        sha256_value = calculate_sha256(filename)
    print(f"{sha256_value}")

    return sha256_value

def calculate_hash(file_path):
    sha1_hash = hashlib.sha1()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(1024)
            if not data:
                break
            sha1_hash.update(data)
    return sha1_hash.hexdigest()



def addnet_hash_safetensors(b):
    """kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def sanitise_path_name(folder_name):
    # Define a regular expression pattern to match invalid characters
    invalid_chars_pattern = re.compile(r'[\\/:"*?<>| ]')

    # Replace invalid characters with an empty string
    sanitised_folder_name = re.sub(invalid_chars_pattern, '', folder_name)

    return sanitised_folder_name

def move_file_to_subfolder(filename, subfolder_name):
    file_location = os.path.dirname(filename)
    destination_folder = os.path.join(file_location, subfolder_name)

    subfolder_name = sanitise_path_name(subfolder_name)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Construct the destination path
    destination = os.path.join(destination_folder, os.path.basename(filename))

    # Handle duplicate filenames
    base, ext = os.path.splitext(destination)
    count = 1
    while os.path.exists(destination):
        destination = f"{base}_{count}{ext}"
        count += 1

    try:
        shutil.move(filename, destination)
    except Exception as e:
        print(f"Error moving '{filename}' to '{destination}': {str(e)}")

def rename_file(filepath, new_filename):
    # Get the directory and base filename
    directory, old_filename = os.path.split(filepath)

    # Split the old filename into base and extension
    old_filename_base, old_filename_ext = os.path.splitext(old_filename)

    # Create the new file path by joining the directory and new base filename with the original extension
    new_filepath = os.path.join(directory, new_filename + old_filename_ext)

    try:
        # Rename the file
        if not os.path.exists(new_filepath):
            os.rename(filepath, new_filepath)
            print(f"File successfully renamed from '{old_filename}' to '{new_filename + old_filename_ext}'")
        else:
            print("filename already exists and will not be over written")
    except FileNotFoundError:
        print(f"Error: File '{old_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

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
                        os.rename(oldfullpath, new_filepath)
                        print(f"File successfully renamed from '{oldfullpath}' to '{new_filepath}'")
                    else:
                        print("filename already exists and will not be over written")
                except FileNotFoundError:
                    print(f"Error: File '{old_filename}' not found.")
                except Exception as e:
                    print(f"An error occurred: {e}")
            #else:
            #    print("filename does not match prefix " + filename)

def get_lora_prompt(lora_path):
    # Open and read the JSON file

    base_name, ext = os.path.splitext(os.path.basename(lora_path))
    file_location = os.path.dirname(lora_path)
    jsonpath = os.path.join(file_location,(base_name + '.json'))
    civitaipath = os.path.join(file_location,(base_name + '.civitai.info'))
    combo = False
    trainedwords = False
    activation_text = False
    nfsw = False
    preferred_weight = False

    json_file_exists = False
    civit_file_exists = False
    if os.path.exists(civitaipath):
        civit_file_exists = True
        with open(civitaipath, 'r', encoding='utf-8') as file:
            data_civitai = json.load(file)
            modelid = data_civitai.get("modelId")
            id = data_civitai.get("id")
            if isinstance(modelid, (int, float)) and isinstance(id, (int, float)):
                combo = str(modelid) + '_' + str(id)
            else:
                print("Invalid modelid or id for " + lora_path)
            
            if create_activation == True:
                trainedwords = data_civitai.get("trainedWords")
                if len(trainedwords) == 0 : 
                    trainedwords = False
                    desc_civitaiini = data_civitai.get("description")
                    if desc_civitaiini != None:
                        print("no trained words in civitai.info.  Does this help ? " + desc_civitaiini)
                        if 'trigger' in desc_civitaiini or 'activation' in desc_civitaiini:
                            print("no trigger found but there is a HIGH probability it is in the description: " + str(desc_civitaiini))
                        else:
                            print("no trigger words and no description in .json")

                    else:
                        print("no trigger words and no description in civitai.info")
                else:
                    trainedwords = [word.strip() for word in trainedwords[0].split(',')]

            nfsw = data_civitai.get("model", {}).get("nsfw", None)
            files = data_civitai.get("files")
            if len(files) == 1:
                size = files[0].get('sizeKB',None)
                result = os.path.getsize(lora_path)/1024
                if size != result:
                    print(f"wrong size ! {lora_path}" )

            if get_checksums == True:
                print("getting checksums")
                pass
                checksums = data_civitai.get("trainedWords")

    else:
        print("no .civitai.info file for " + lora_path)

    if create_activation == True:
        if os.path.exists(jsonpath):
            json_file_exists = True
            with open(jsonpath, 'r', encoding='utf-8') as file:
                data_json = json.load(file)
                # Extract the required fields from the JSON data_json
                preferred_weight = data_json.get("preferred weight", 1)
                activation_text = data_json.get("activation text")
                if activation_text == '' or activation_text == None :
                    activation_text = False

                    desc_json = data_json.get("description")
                    if desc_json != None:
                        desc_json = desc_json.lower()
                        if 'trigger' in desc_json or 'activation' in desc_json:
                            print("no trigger found but there is a HIGH probability it is in the description: " + str(desc_json))
                        else:
                            print("no trigger words and no description in .json")

                else:
                    activation_text = [word.strip() for word in activation_text.split(',')]

                try:
                    if float(preferred_weight) == 0:
                        preferred_weight = 1
                except:
                    preferred_weight = 1
        else:
            print("no json file for " + lora_path)

    if create_activation == True:
        if trainedwords == False and activation_text == False:
            print(lora_path + " no activation words")
            print(lora_path + " no activation words")
        
    #lora_name = Path(lora_path).stem

    return combo,trainedwords,activation_text,nfsw,preferred_weight,

def main():
    global root_directory
    root_directory = os.path.normpath(root_directory)


    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            if filename.endswith('.safetensors'):
                fullpath = os.path.normpath(os.path.join(root, filename))
                ret,actwrd1,actwrd2,nsfw,weight = get_lora_prompt(fullpath)
                
                if ret != False and rename_Loras == True:
                    rename_file_grouping(fullpath,ret)
                    #time.sleep(4)
                elif rename_Loras == False:
                    print("renaming Lora disabled")
                elif ret == False:
                    print("Not enough information to rename Lora")
                else:
                    print(f"could not process the file {fullpath}.  No Civitai modelid  ")
                    continue
                #print ("actwrd1: " + str(type(actwrd1)) + "actwrd2: " + str(type(actwrd2)))
                if actwrd1 != False and actwrd2 != False:
                    unique_elements = list(set(actwrd1 + actwrd2))
                elif actwrd2 != False:
                    unique_elements = actwrd2
                elif actwrd1 != False:
                    unique_elements = actwrd1
                else:
                    print("no activation text")
                    unique_elements = ['NOTRIGGER']
                #write_to_log(log_file,f"{fullpath},{unique_elements}")
                    
                    wrapped_string = f'"{",".join(unique_elements)}"'

                if create_activation == True:
                    write_to_log(Trigger_file,f"{fullpath},{wrapped_string},{nsfw},{weight}")



root_directory = '/file/to/sort/'
useapikey = True
rename_Loras = False
create_activation = False
get_checksums = True

if useapikey == True:
    #unused here
    apifile = os.path.join(get_script_path(),"apikey.py")
    if os.path.exists(apifile):
        import apikey

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


debug = True
log_file = os.path.join(get_script_path(),get_script_name() + '.log')
Trigger_file = os.path.join(get_script_path(),'trigger.txt')
main()