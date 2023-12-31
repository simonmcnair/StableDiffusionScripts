import os
import shutil
from PIL import Image
from pathlib import Path

import re

def get_script_name():
    # Use os.path.basename to get the base name (script name) from the full path
    #basename = os.path.basename(path)
    return Path(__file__).stem
    #return os.path.basename(__file__)

def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))

def sanitize_folder_name(folder_name):
    # Define a regular expression pattern to match invalid characters
    invalid_chars_pattern = re.compile(r'[\\/:"*?<>|]')

    # Replace invalid characters with an empty string
    sanitized_folder_name = re.sub(invalid_chars_pattern, '', folder_name)

    return sanitized_folder_name


# Function to move a file to a subfolder with the given name
def move_file_to_subfolder(filename, subfolder_name):
    file_location = os.path.dirname(filename)
    destination_folder = os.path.join(file_location, subfolder_name)

    subfolder_name = sanitize_folder_name(subfolder_name)
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

def move_file_to_fixedfolder(filename, folder,keyword):
    keyword = sanitize_folder_name(keyword)

    file_location = os.path.dirname(filename)
    destination_folder = os.path.join(file_location, folder,keyword)

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

# Search for files containing a specific string
def search_and_move_files(directory, search_string, foldername,dest=False, ):
    movetosubfolder = False
    movetofixedfolder = False

    if isinstance(search_string, str):
        print("It's a string!")
        if ',' in search_string:
           terms = search_string.split(',')
        else:
           terms = list(search_string)
    elif isinstance(search_string, dict):
        print("It's a dictionary!")
        return None
        # Perform dictionary-related actions
    elif isinstance(search_string, list):
        print("It's a list!")
        terms = search_string

    if dest == False:
        movetosubfolder = True
    else:
        movetofixedfolder = True

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.isfile(file_path):
                continue
            hasparameters = False
            if file_path.endswith(".png"):
                with Image.open(file_path) as img:
                    try:
                        parameter = img.info.get("parameters")
                        if parameter is not None:
                            print(file_path + " has metadata.")
                            hasparameters = True
                            parameter = parameter.lower()

                        else:
                            print("PNG with no metadata")
                            badfile = True
                    except:
                        badfile = True

            if hasparameters ==True:
                if any(terms) in parameter:
                    print(parameter)
                    print(f"Found '{search_string}' in: {file_path}")
                    #user_input = input("Do you want to move this file? (y/n): ").strip().lower()
                    #if user_input == 'y':
                    if movetosubfolder == True:
                        move_file_to_subfolder(file_path, foldername)
                    if movetofixedfolder ==True:
                        move_file_to_fixedfolder(file_path,dest ,foldername)
                else:
                    print(search_string + " not found in " + file_path)

            else:
                continue

            #with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            #    contents = f.read()

            #if search_string in contents:
                

# Directory to search
search_directory =  '/path/to/search'
destination =  '/folder/to/move/to'
search_string = 'searchterm'
search_term_folder = 'foldertogroupunder'

apifile = os.path.join(get_script_path(), "apikey.py")
if os.path.exists(apifile):
    exec(open(apifile).read())
    api_key = apikey
    print("API Key:", api_key)
else:
    print("apikey.py not found in the current directory.")

localoverridesfile = os.path.join(get_script_path(), "localoverridesfile_" + get_script_name() + '.py')

if os.path.exists(localoverridesfile):
    exec(open(localoverridesfile).read())
    #api_key = apikey
    #print("API Key:", api_key)
else:
    print("No local overrides.")

search_and_move_files(search_directory, search_string, search_term_folder, destination)
