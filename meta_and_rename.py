import os
import shutil
from PIL import Image
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

path = cwd = os.getcwd()
path = "c:\\users\\simon\\Downloads\\stable-diffusion\\consolidated"
#for filename in os.listdir("."):
for root, dirs, files in os.walk(path):
    for filename in files:
        item_path = os.path.join(root, filename)

        if os.path.isfile(item_path):
        
            badfile = False
            hasparameters = False
            parameter = None
            model = None
            seed = None

            if filename.endswith(".png"):
                with Image.open(item_path) as img:
                    try:
                        parameter = img.info.get("parameters")
                        if parameter is not None:
                            print(filename + " has metadata.")
                            hasparameters = True
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

            if hasparameters==True:
                model = getmodel(parameter)

                seed = getseed(parameter)
                datetimemod = get_sanitized_download_time(item_path)

                if '<lora:' in parameter:
                    # Use a regular expression to find all words between lora: and :?>
                    matches = re.findall(r'lora:(.*?):', parameter)
                    Loras = '_'.join(matches)


                    if model is not None and seed is not None:
                        new_filename = model + '_' + seed + '_' + datetimemod + '_Loras_' + Loras + os.path.splitext(filename)[1]
                        new_item_path = os.path.join(root, new_filename)

                        print(new_item_path)

                        try:
                            if not os.path.samefile(item_path, new_item_path):
                                    shutil.move(item_path, new_item_path)
                        except Exception as e:
                            print(str(e))
                    elif seed is not None:
                        new_filename = seed + '_' + datetimemod + '_Loras_' + Loras + os.path.splitext(filename)[1]
                        new_item_path = os.path.join(root, new_filename)

                        print(new_item_path)

                        try:
                            if not os.path.samefile(item_path, new_item_path):
                                    shutil.move(item_path, new_item_path)
                        except Exception as e:
                            print(str(e))

                    else:
                        print("seed or model is blank.  Skipping")
                else:
                    print("uses no Loras")

                    if model is not None and seed is not None:
                        new_filename = model + '_' + seed + '_' + datetimemod + os.path.splitext(filename)[1]
                        new_item_path = os.path.join(root, new_filename)

                        print(new_item_path)

                        try:
                            if not os.path.samefile(item_path, new_item_path):
                                    shutil.move(item_path, new_item_path)
                        except Exception as e:
                            print(str(e))
                    elif seed is not None:
                        new_filename = seed + '_' + datetimemod + '_Loras_' + Loras + os.path.splitext(filename)[1]
                        new_item_path = os.path.join(root, new_filename)

                        print(new_item_path)

                        try:
                            if not os.path.samefile(item_path, new_item_path):
                                    shutil.move(item_path, new_item_path)
                        except Exception as e:
                            print(str(e))
                    else:
                        print("Parameters but no seed or model")
            else:
                if 'nometa' not in item_path:
                    print("should this be in this folder ?")
                else:
                    print("no parameters")




