import os
import shutil
from PIL import Image


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
path = "c:\\users\\simon\\Downloads\\stable-diffusion\\consolidated"
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
                print(filename + " has metadata.  Moving to meta subdirectory")
                move_to_subfolder(item_path,"meta")
