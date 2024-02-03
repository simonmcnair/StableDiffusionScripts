import os
import shutil
from PIL import Image
from pathlib import Path
import platform

import re

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

def sanitize_folder_name(folder_name):
    # Define a regular expression pattern to match invalid characters
    invalid_chars_pattern = re.compile(r'[\\/:"*?<>|]')

    # Replace invalid characters with an empty string
    sanitized_folder_name = re.sub(invalid_chars_pattern, '', folder_name)

    return sanitized_folder_name


# Function to move a file to a subfolder with the given name
def move_file_to_subfolder(filepath, subfolder_name):
    file_location = os.path.dirname(filepath)
    destination_folder = os.path.join(file_location, subfolder_name)

    subfolder_name = sanitize_folder_name(subfolder_name)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Construct the destination path
    destination = os.path.join(destination_folder, os.path.basename(filepath))

    # Handle duplicate filenames
    base, ext = os.path.splitext(destination)
    count = 1
    while os.path.exists(destination):
        destination = f"{base}_{count}{ext}"
        count += 1

    if os.path.abspath(filepath) == os.path.abspath(destination):
        print("Source and destination are the same. No move needed.")
        return

    try:
        shutil.move(filepath, destination)
    except Exception as e:
        print(f"Error moving '{filepath}' to '{destination}': {str(e)}")

def move_file_to_fixedfolder(filepath, folder,keyword):
    keyword = sanitize_folder_name(keyword)

    file_location = os.path.dirname(filepath)
    destination_folder = os.path.join(file_location, folder,keyword)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Construct the destination path
    destination = os.path.join(destination_folder, os.path.basename(filepath))

    # Handle duplicate filenames
    base, ext = os.path.splitext(destination)
    count = 1
    while os.path.exists(destination):
        destination = f"{base}_{count}{ext}"
        count += 1

    if os.path.abspath(filepath) == os.path.abspath(destination):
        print("Source and destination are the same. No move needed.")
        return

    try:
        shutil.move(filepath, destination)
    except Exception as e:
        print(f"Error moving '{filepath}' to '{destination}': {str(e)}")

def find_folder_for_term(search_data,term):
    for entry in search_data:
        if term in entry["terms"]:
            return entry["folder"], entry["Move_to_subfolder"]
    return None  # Return None if the term is not found in any entry


def search_and_move_files(search_term_array,foldertoSearch):

    all_search_terms = [term for entry in search_term_array for term in entry["terms"]]

      
    print("searching " + foldertoSearch)



    for root, dirs, files in os.walk(foldertoSearch):
        for file in files:
            print("processing " + file)
            file_path = os.path.join(root, file)
            if not os.path.isfile(file_path):
                continue
            hasparameters = False
            found = False
            checkfilename = False
            for search_term in all_search_terms:
                if search_term.lower() in file:
                    print('search term exists in filename')

                    keyword, foldername = find_folder_for_term(search_term_array,search_term)

                    if keyword != False:
                        movetofixedfolder = True
                    else:
                        movetofixedfolder = False

                    if checkfilename == True:
                        if movetofixedfolder == True:
                            move_file_to_fixedfolder(file_path,foldername,keyword)
                            found = True
                            continue
                        else:
                            move_file_to_subfolder(file_path, foldername)
                            found = True
                            continue
            
            if found == False:
                print("Terms did not exist in filename.  Checking params for " + file)

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
                    #if any(terms) in parameter:
                    for search_term in all_search_terms:
                    #if any(term in parameter for term in terms):
                        if search_term.lower() in parameter:
                            print(parameter)
                            print(f"Found '{search_term}' in parameters for : {file_path}")

                            keyword, foldername = find_folder_for_term(search_term_array,search_term)

                            if keyword != False:
                                movetofixedfolder = True
                            else:
                                movetofixedfolder = False

                            #user_input = input("Do you want to move this file? (y/n): ").strip().lower()
                            #if user_input == 'y':

                            if movetofixedfolder == True:
                                move_file_to_fixedfolder(file_path ,foldername,keyword)
                            else:
                                move_file_to_subfolder(file_path, foldername)

                        #else:
                        #    print(search_term + " not found in parameters for " + file_path)

                else:
                    print("no parameters.  Skipping")
                    continue


# Directory to search
search_directory =  '/path/to/search'
destination =  '/folder/to/move/to'

#move_to_subfolder should be the directory you want it moving to.  If set to False it will put it in a subdirectory of the current folder
search_data = [
    {"terms": ['searchforthis'],
     "folder": "move searchforthis to here",
     "Move_to_subfolder": destination},
    {"terms": ["searchforthistoo"],
     "folder": "move searchforthistoo to here",
     "Move_to_subfolder": destination},
    {"terms": ['also search for this'],
     "folder": 'move also search for this to here',
     "Move_to_subfolder": destination}
    # Add more search terms and folders as needed
]

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



search_and_move_files(search_data,search_directory)
