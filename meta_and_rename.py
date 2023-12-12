import os
import shutil
from PIL import Image
from PIL.PngImagePlugin import PngInfo

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


def move_to_subfolder(path, subfolder):
    # Check if the path is a directory or a file

    last_folder = os.path.basename(os.path.dirname(path))

    if subfolder in last_folder:
        print(path + " directory tree already contains " + subfolder)
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
                dest_file = os.path.join(subfolder_path, file)
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
path = "c:\\users\\simon\\Downloads\\stable-diffusion\\consolidated\\Sort"
#for filename in os.listdir("."):
for root, dirs, files in os.walk(path):
    for filename in files:
        item_path = os.path.join(root, filename)

        if os.path.isfile(item_path):
        
            badfile = False
            hasparameters = False

            if filename.endswith(".png"):
                with Image.open(item_path) as img:
                    try:
                        parameter = img.info.get("parameters")
                        if parameter is not None:
                            print(filename + " has metadata.")
                        else:
                            print("PNG with no metadata")
                            badfile = True
                    except:
                        badfile = True
            elif filename.endswith(".jpeg") or filename.endswith(".jpg"):
                badfile = True
            else:
                print("Ignoring unsupported filetype: " + filename)
                continue

            if badfile==True:
                print(filename + " has no metadata.  Moving to nometa subdirectory")
                move_to_subfolder(item_path,"nometa")
            else:

                model = ""
                seed = ""
                Loras = ""
                new_filename = ""

                model = getmodel(parameter)
                seed = getseed(parameter)

                if model is not None:
                    new_filename = model + '_'
                else:
                    new_filename = "nomodel_"

                if seed is not None:
                    new_filename = new_filename + seed  + '_'
                else:
                    new_filename = new_filename + "noseed_"


                new_filename = new_filename + '_' + get_sanitized_download_time(item_path) + '_'
                # os.path.splitext(filename)[1]

                if 'lora:' in parameter:
                    # Use a regular expression to find all words between lora: and :?>

                    if "Negative prompt" in parameter:
                        parts = parameter.split("Negative prompt", 1)
                    else:
                        parts = re.split(r'[\r\n]+', parameter)
                        #parts = re.split(r'[\r\n]Steps', parameter)

                    if len(parts) > 1:
                        matches = re.findall(r'lora:(.*?):', parts[0])
                        Loras = '_'.join(matches)

                        tags = findtags(parts[0])
                    else:
                        print("Prompt me")


                    new_filename = new_filename + 'Loras_' + Loras + '_'
                else:
                    print("uses no Loras")

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
