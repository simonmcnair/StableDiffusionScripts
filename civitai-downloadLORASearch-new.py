import requests
import time
import os
import json
import re
from tqdm import tqdm


from pathlib import Path

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
    qty = 10
    togglequit = False

#    modelsearch = f'https://civitai.com/api/v1/models?tag={DownloadLORASearch}&limit={qty}'
#    modeltag = f'https://civitai.com/api/v1/models?limit={qty}&types=LORA&query={DownloadLORASearch}'

    modelsearch = f'https://civitai.com/api/v1/models?tag={DownloadLORASearch}&limit={qty}&types=LORA'
    modeltag = f'https://civitai.com/api/v1/models?limit={qty}&types=LORA&query={DownloadLORASearch}'

    myarray = []
    myarray.append(modelsearch)
    myarray.append(modeltag)

    for eachsearch in myarray:
        i = 0
        page = 1
        while True:
            print(f"URL is {eachsearch}.  Page is {page}")

            headers = {}
            headers['Content-Type'] =  'application/json'

            params = {'page': page}
            while True:
                try:
                    response = requests.get(eachsearch,headers=headers, params=params)
                except Exception as e:
                    write_to_log(logfile_path, "Error " + str(e))

                if response.status_code == 200:
                    try:
                        data = response.json()
                    except Exception as e:
                        write_to_log(logfile_path, "Error " + str(e))
                    break
                elif response.status_code == 401 and api_key:
                    # Retry the request with the API key
                    headers["Authorization"] = f"Bearer {api_key}"
                elif "be back in a few minutes" in str(response.content):
                    print("error.  Site down")
                    exit()    
                else:
                    time.sleep(5)
                    write_to_log(logfile_path, "status code: " + str(response.status_code) + " " + response.reason)

            # Check if there are models in the response
            if 'items' in data:
                # Extract 'id' field from each model and add it to the list
                totalcnt = data['metadata'].get('totalItems')
                write_to_log(logfile_path, "totalcnt = " + str(totalcnt))
                write_to_log(logfile_path, "page = " + str(data['metadata'].get('currentPage')) + " of #" + str(data['metadata'].get('totalPages')))
                for eachitem in data['items']:
                    i += 1
                    id = eachitem.get('id')
                    name = eachitem.get('name')
                    write_to_log(logfile_path, "processing #" + str(i) + " of " + str(totalcnt) + " " + str(id) + f" ({name})")

                    for submodel in eachitem['modelVersions']:
                        model = submodel.get('name')
                        model_id = submodel.get('id')
                        for file in submodel['files']:
                            if file.get('type') =="Model":
                                write_to_log(successfile_path, f"found submodel LORA: {model}")
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

                                filesize_should_be = file.get('sizeKB')
                                fileexists = False
                                faileddownload = False

                                if os.path.exists(download_fullpath):
                                    filesize_is = os.path.getsize(download_fullpath)
                                    fileexists = True
                                    write_to_log(logfile_path, "File already exists")

                                max_retries = 3
                                download = True

                                for retry in range(max_retries):
                                    if download == True:
                                        if retry != 0:
                                            print(f"retrying: {retry}")
                                        with requests.get(downloadurl, headers=headers, stream=True) as response:
                                            if response.status_code == 200:
                                                # The request was successful
                                                file_size = int(response.headers.get("content-length", 0))
            
                                                if file_size != filesize_is:
                                                    faileddownload = True
                                                    write_to_log(logfile_path, f"File is wrong size.  {filesize_is} and should be {file_size}.   Failed download.")
                                                else:
                                                    write_to_log(logfile_path, f"File is correct size ({filesize_is}).  Next..")
                                                    download = False
                                                    continue
                                                    
                                                write_to_log(logfile_path, "downloading " + downloadurl + ".  Filename: " + download_filename + ". size: " + str(filesize_should_be))


                                                with open(download_fullpath, "wb") as file, tqdm(
                                                    desc="Downloading",
                                                    total=file_size,
                                                    unit="B",
                                                    unit_scale=True,
                                                    unit_divisor=1024,
                                                ) as bar:
                                                    # Iterate over the content in chunks and write to the file
                                                    for chunk in response.iter_content(chunk_size=8192):  # You can adjust the chunk size
                                                        if chunk:
                                                            #print("chunk")
                                                            file.write(chunk)
                                                            bar.update(len(chunk))
                                                print("Download complete.")
                                                dump_to_json(submodel,downloadJSON_fullpath)
                                                write_to_log(logfile_path, f"File downloaded successfully to {download_fullpath}")
                                                write_to_log(successfile_path, f"Model: {model}.  URL: {downloadurl}.  Filename: {download_filename}")
                                                download = False
                                                continue

                                            elif response.status_code == 401 and api_key:
                                                # Retry the request with the API key
                                                headers["Authorization"] = f"Bearer {api_key}"
                                                print("request denied.  Adding auth header")
                                            else:
                                                # The request was not successful, handle the error
                                                write_to_log(logfile_path, f"Failed to download file. Status code: {response.status_code}")

                            else:
                                write_to_log(logfile_path, "file type is" + f": {file.get('type')}.  Filename: {file.get('name')}")
            else:
                print("no items returned")

            #if togglequit == True:
            # break
            # Check if there are more pages
            if data['metadata'].get('currentPage') >= (data['metadata'].get('totalPages')):
                #print("lastpage")
                #togglequit = True
                break
            else:
                page += 1 
    print("Finished")
Lora_download_to = '/folder/to/download/to'
DownloadLORASearch = 'Lora to search for'
api_key = 'void'
download_types = []
download_types += ('Checkpoint', 'TextualInversion', 'MotionModule','Hypernetwork', 'AestheticGradient', 'LORA', 'LoCon','Controlnet', 'Upscaler','VAE','Poses','Wildcards','Other')

apifile = os.path.join(get_script_path(), "apikey.py")


try:
    from apikey import api_key
    print("apikey found" + api_key)
except ImportError:
    print("apikey.py not found in the current directory.")


localoverridesfile = os.path.join(get_script_path(), "localoverridesfile_" + get_script_name() + '.py')

if os.path.exists(localoverridesfile):
    exec(open(localoverridesfile).read())
else:
    print("No local overrides.")

logfile_path = os.path.join(Lora_download_to,'logfile.log')

successfile_path = os.path.join(Lora_download_to,'successfile.log')

get_models()
