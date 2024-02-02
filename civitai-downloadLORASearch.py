import requests
import time
import os
import json
import re
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

def sanitise_filepath(filepath):
    # Define the set of invalid characters in Windows and Linux file paths
    filepath = filepath.replace('\\','/')
    invalid_characters = set(['<', '>', '"', '\\','|', '?', '*',' '])

    # Replace or remove invalid characters
    sanitised_filepath = ''.join(char if char not in invalid_characters else '_' for char in filepath)

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

def get_models():

    global api_key
    global DownloadLORASearch
    # Initialize the first page
    page = 1
    qty = 10
    i = 0
    togglequit = False
    while True:
        # Make API request for the current page
        headers = {'Content-Type': 'application/json'}
        params = {'page': page}

        while True:
            try:
                est = f'https://civitai.com/api/v1/models?limit={qty}&types=LORA&query={DownloadLORASearch}'
                response = requests.get(f'https://civitai.com/api/v1/models?limit={qty}&types=LORA&query={DownloadLORASearch}', headers=headers, params=params)
            except Exception as e:
                 write_to_log(logfile_path, "Error " + str(e))

            if "be back in a few minutes" in str(response.content):
                print("error.  Site down")
                exit()

            if response.status_code == 401 and api_key:
                # Retry the request with the API key
                headers["Authorization"] = f"Bearer {api_key}"
                response = requests.get(f'https://civitai.com/api/v1/models?limit=10&types=LORA&query={DownloadLORASearch}', headers=headers, params=params)

                
            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception as e:
                    write_to_log(logfile_path, "Error " + str(e))
                break
            else:
                 time.sleep(5)
                 write_to_log(logfile_path, "status code: " + str(response.status_code) + " " + response.reason)
 
        # Check if there are models in the response
        if 'items' in data:
            # Extract 'id' field from each model and add it to the list
            totalcnt = data['metadata'].get('totalItems')
            write_to_log(logfile_path, "totalcnt = " + str(totalcnt))
            write_to_log(logfile_path, "page = " + str(data['metadata'].get('currentPage')) + " of #" + str(data['metadata'].get('totalPages')))
            for each in data['items']:
                i += 1
                id = each.get('id')
                name = each.get('name')
                write_to_log(logfile_path, "processing #" + str(i) + " of " + str(totalcnt) + " " + str(id) + f" ({name})")

                for each1 in each['modelVersions']:
                    model = each1.get('name')
                    model_id = each1.get('id')
                    for file in each1['files']:
                        if file.get('type') =="Model":
                            write_to_log(successfile_path, "found LORA")
                            downloadurl = file.get('downloadUrl')
                            unused1 = str(file.get('id'))
                            lorafilename = file.get('name')
                            destination_folder = sanitise_filepath(os.path.join(Lora_download_to,DownloadLORASearch))
                            #downloadfilename = name + '_' + model + '_' + file.get('name')
                            download_filename = f"{id}_{model_id}_{lorafilename}"
                            download_fullpath = sanitise_filepath(os.path.join(destination_folder,download_filename))

                            downloadJSON_filename = f"{id}_{model_id}_{lorafilename}.json"
                            downloadJSON_fullpath = sanitise_filepath(os.path.join(destination_folder,downloadJSON_filename))

                            if not os.path.exists(destination_folder):
                                os.makedirs(destination_folder)

                            if not os.path.exists(download_fullpath):
                                write_to_log(logfile_path, "downloading " + downloadurl + ".  Filename: " + download_filename + ". size: " + str(file.get('sizeKB')))
                                try:
                                    response = requests.get(downloadurl)
                                except Exception as e:
                                    write_to_log(logfile_path, "Error " + str(e))

                                if response.status_code == 401 and api_key:
                                    # Retry the request with the API key
                                    headers["Authorization"] = f"Bearer {api_key}"
                                    response = requests.get(downloadurl, headers=headers)

                                if response.status_code == 200:
                                    with open(download_fullpath, 'wb') as file2:
                                        file2.write(response.content)

                                    dump_to_json(each1,downloadJSON_fullpath)
                                    write_to_log(logfile_path, f"File downloaded successfully to {download_fullpath}")
                                    write_to_log(successfile_path, f"{model},{downloadurl},{download_filename}")
                                else:
                                    write_to_log(logfile_path, f"Failed to download file. Status code: {response.status_code}")
                            else:
                                write_to_log(logfile_path, "file already exists")
                        else:
                            write_to_log(logfile_path, "file type is" + f": {file.get('type')}.  Filename: {file.get('name')}")
        else:
            print("no items returned")

        if togglequit == True:
            break
        # Check if there are more pages
        if data['metadata'].get('currentPage') == (data['metadata'].get('totalPages')):
            print("lastpage")
            togglequit = True
        else:
            page += 1 

Lora_download_to = '/folder/to/download/to'
DownloadLORASearch = 'Lora to search for'


apifile = os.path.join(get_script_path(), "apikey.py")
if os.path.exists(apifile):
    exec(open(apifile).read())
    api_key = apikey
    print("API Key:", api_key)
else:
    print("apikey.py not found in the current directory.")

current_os = get_operating_system()

if current_os == "Windows":
    print("Running on Windows")
elif current_os == "Linux":
    print("Running on Linux")

localoverridesfile = os.path.join(get_script_path(), "localoverridesfile_" + get_script_name() + '_' + current_os + '.py')

if os.path.exists(localoverridesfile):
    exec(open(localoverridesfile).read())
    #api_key = apikey
    #print("API Key:", api_key)
    print("local override file is " + localoverridesfile)

else:
    print("local override file would be " + localoverridesfile)



logfile_path = os.path.join(Lora_download_to,'logfile.log')

successfile_path = os.path.join(Lora_download_to,'successfile.log')

get_models()
