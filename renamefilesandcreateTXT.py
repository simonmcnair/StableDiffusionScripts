import os
from PIL import Image

import shutil

def move_to_subfolder(path, subfolder):
    # Check if the path is a directory or a file
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

def extract_text_after(list_obj, text):
    for element in list_obj:
        if text in element:
            return element.split(text)[-1].strip()
    return None

def extract_text_after2(list_obj, text):
    for element in list_obj:
        if element.strip().startswith(text + ":"):
            return element.split(":")[-1].strip()
    return None


for filename in os.listdir("."):
    badfile = False
    if filename.endswith(".png"):
        with Image.open(filename) as img:
            parameter = img.info.get("parameters")
            if parameter is not None:
                print(filename + " has metadata.")

                test = parameter.split(",")
                result = extract_text_after2(test,"Model")
                if None is None:
                    print("No Model specified in " + filename)
                else:
                    output_filename = result + "_" + os.path.splitext(filename)[0] + ".txt"
                    with open(output_filename, "w", encoding="utf-8") as output_file:
                        output_file.write(parameter)
                        #output_file.write(parameter)
            else:
                badfile = True

    if badfile==True:
        print(filename + " has no metadata.  Moving")
        move_to_subfolder(filename,"nometa")
    if badfile==False:
        if result is not None:
            print(filename + " metadataextracted.  Moving")
            shutil.move(filename, result+ "_" + filename)

       
