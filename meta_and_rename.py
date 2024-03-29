import os
import shutil
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
import re
from datetime import datetime

def extract_text_after2(list_obj, text):
    for element in list_obj:
        if element.strip().lower().startswith(text.lower() + ":"):
            return element.split(":")[-1].strip()
    return None

def getmodel(mystr):
    test = parameter.split(",")
    try:
        res = extract_text_after2(test,"Model")
    except:
        return None
    return res

def getseed(mystr):
    test = parameter.split(",")
    try:
        res = extract_text_after2(test,"Seed")
    except:
        return None
    return res

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
    re_hypernet_hash = re.compile(r"\(([0-9a-f]+)\)$")
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

def findtags(inputstring):
    inputstring = inputstring.replace("\r", "")
    inputstring = inputstring.replace("\n", "")
    inputstring = [substring.strip() for substring in inputstring.split(',')]

    
    print(inputstring)
    return

def get_sanitized_download_time(filepath):
    # Get the modification time of the file
    mtime = os.path.getmtime(filepath)
    # Convert the modification time to a datetime object
    dt = datetime.fromtimestamp(mtime)
    # Format the datetime as a string in the format YYYY-MM-DD_HH-MM-SS
    dt_string = dt.strftime("%Y-%m-%d_%H-%M-%S")
    # Replace any spaces or colons with underscores
    sanitized_dt_string = dt_string.replace(" ", "_").replace(":", "_")
    # Return the sanitized datetime string
    return sanitized_dt_string

def trim_filepath(filepath):
    # Get the maximum length allowed for a path on Windows
    max_path_length = 260

    # Get the length of the root directory (e.g., C:\)
    root_length = len(os.path.splitdrive(filepath)[0]) + 2  # Add 2 for the drive letter and colon

    # If the path is already within the limit, return it as is
    if len(filepath) <= max_path_length:
        return filepath

    # Calculate the maximum length for the remaining part of the path
    max_remaining_length = max_path_length - root_length

    # Trim the remaining part of the path if it exceeds the maximum length
    remaining_path = os.path.splitdrive(filepath)[1]
    if len(remaining_path) > max_remaining_length:
        # If the path is longer than the maximum length, keep the basename and truncate the rest
        basename = os.path.basename(filepath)
        remaining_length = max_remaining_length - len(basename)
        dir_path = os.path.dirname(filepath)
        trimmed_path = os.path.join(dir_path[:remaining_length], basename)
        return trimmed_path
    else:
        return filepath
    
def remove_non_english(text):
    # Define a regular expression pattern to match English letters and spaces
    #english_pattern = re.compile("[^a-zA-Z0-9\s!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]")
    english_pattern = re.compile(r"[^a-zA-Z0-9\s!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]")

    # Use the sub() method to replace non-English characters with an empty string
    cleaned_text = english_pattern.sub("", text)

    return trim_filepath(cleaned_text)

def has_parameters(filepath, verify="false"):

    if os.path.exists(filepath) and os.path.isfile(filepath):
        if filepath.endswith(".png"):
            try:
                with Image.open(filepath) as img:
                    try:
                        parameter = img.info.get("parameters")
                        if parameter is not None:
                            print(filepath + " has metadata.")
                            if verify == "basic":
                                res = parse_generation_parameters(parameter)
                                if res != None:
                                    if 'Prompt' in res:
                                        print("has basic metadata")
                                        return res, True
                                    else:
                                        print("insufficient parameters to be considered a prompt")
                            elif verify == 'full':
                                res = parse_generation_parameters(parameter)
                                #verify just checks for more stuff
                                if 'Prompt' in res and 'Seed' in res and 'Sampler' in res:
                                    #highlevel looks valid
                                    return res, True
                                else:
                                    print("insufficient parameters to be considered a prompt")
                        else:
                            print("PNG with no metadata")
                    except Exception as e:
                        print("damaged png file")

            except Exception as e:
                print(f"Error {e}")
        else:
            print("non png files don't have parameters")
    if verify == "full" or verify == "basic":
        return None, False
    else:
        return False

def getloras(parameter):

    matches = re.findall(r'<lora:(.*?):', parameter)
    loras = '_'.join(set(matches))

    #loras = '_'.join(matches)

    if len(loras) >0:
        #print("Lora !")
        return loras
    else:
        return None

def move_to_subfolder(path, subfolder):
    # Check if the path is a directory or a file

    #last_folder = os.path.basename(os.path.dirname(path))

    #folders = path.split(os.path.sep)

    if any(subfolder.lower() == folder.lower() for folder in path.split(os.path.sep)):

    # Iterate through each folder and run the custom function
    #for folder in folders:
    #    if subfolder.lower() == folder.lower():
            print(path + " directory tree already has a folder containing " + subfolder)
            return

    if os.path.isdir(path):
        # Create the subfolder if it doesn't exist
        subfolder_path = os.path.join(path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Move all files in the directory to the subfolder
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                base_name, ext = os.path.splitext(file)
                dest_file = os.path.join(subfolder_path, file + ext)
                count = 1
                while os.path.exists(dest_file):
                    new_name = f"{base_name}_{count}{ext}"
                    dest_file = os.path.join(subfolder_path, new_name)
                    count += 1
                shutil.move(os.path.join(path, file), dest_file)

    elif os.path.isfile(path):
        # Create the subfolder if it doesn't exist
        subfolder_path = os.path.join(os.path.dirname(path), subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Move the file to the subfolder
        base_name, ext = os.path.splitext(os.path.basename(path))
        dest_file = os.path.join(subfolder_path, os.path.basename(path))
        count = 1
        while os.path.exists(dest_file):
            new_name = f"{base_name}_{count}{ext}"
            dest_file = os.path.join(subfolder_path, new_name)
            count += 1
        shutil.move(path, dest_file)


path = cwd = os.getcwd()
onlyprocess_png = True
path = 'X:\\dif\\stable-diffusion-webui-docker\\output\\txt2img'
#for filename in os.listdir("."):
for root, dirs, files in os.walk(path):
    for filename in files:
        item_path = os.path.join(root, filename)

        if os.path.isfile(item_path):
            if onlyprocess_png == True and not item_path.endswith('.png'):
                continue

            hasparameters = False
            parameter, result = has_parameters(item_path, "basic")
            if result:
                print("has parameters")
            else:
                print(filename + " has no metadata.  Moving to nometa subdirectory")
                move_to_subfolder(item_path,"nometa")
                continue

            model = ""
            seed = ""
            Loras = ""
            new_filename = ""

            if 'Model' in parameter:
                model = parameter['Model']
            if 'Seed' in parameter:
                seed = parameter['Seed']

            if model is not None and model != '':
                new_filename = model + '_'
            else:
                new_filename = "nomodel_"

            if seed is not None and seed != '':
                new_filename = new_filename + seed  + '_'
            else:
                new_filename = new_filename + "noseed_"

            new_filename = new_filename + get_sanitized_download_time(item_path) + '_'
            # os.path.splitext(filename)[1]

            if 'lora' in str(parameter).lower():
                Loras = getloras(parameter['Prompt'])
                if Loras != None:
                    foundlora = True
                    print(f"loras: {Loras}")
                    new_filename = new_filename + 'Loras_' + Loras + '_'
            else:
                print("uses no Loras")

            new_filename = remove_non_english(new_filename)
            new_filename = new_filename + os.path.splitext(filename)[1]
            new_item_path = os.path.join(root, new_filename)

            print(new_item_path)

            if item_path not in new_item_path:
                try:
                    shutil.move(item_path, new_item_path)
                except Exception as e:
                    print(str(e))
            else:
                print("doesn't need moving.  Src and dest are the same: " + item_path + ' ' + new_item_path)


            print(new_item_path + " has metadata.  Moving to meta subdirectory")
            move_to_subfolder(new_item_path,"meta")
