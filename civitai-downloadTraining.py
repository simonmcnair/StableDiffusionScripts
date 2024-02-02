import requests
import time
import os
from pathlib import Path
import json
import platform

def get_operating_system():
    system = platform.system()
    return system

def dump_to_json(data, filename):
    """
    Dump a Python object to a JSON file.

    Parameters:
    - data: Python object to be dumped to JSON.
    - filename: Name of the JSON file to be created.
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)  # indent for pretty formatting (optional)

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

def get_models():

    global api_key
    # Initialize the first page
    page = 1
    batchsize = 100

    while True:
        # Make API request for the current page
        headers = {'Content-Type': 'application/json'}

#        if api_key:
#            headers["Authorization"] = f"Bearer {api_key}"

        params = {'page': page}
        print("processing page " + str(page))
        u = 0
        while True:
            try:
                req = f'https://civitai.com/api/v1/models?limit={batchsize}&types=LORA'
                response = requests.get(req, headers=headers, params=params)
            except Exception as e:
                 write_to_log(logfile_path, "Error " + str(e))

            if "be back in a few minutes" in str(response.content):
                print("error.  Site down")
                exit()

            if response.status_code == 404:
                print("not found")
                break
            if response.status_code == 401 and api_key:
                # Retry the request with the API key
                headers["Authorization"] = f"Bearer {api_key}"
                response = requests.get(req, headers=headers, params=params)

            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'items' in data:
                        print("found items.  Legitimate data")
                        break
                except Exception as e:
                    write_to_log(logfile_path, "Error " + str(e))
                
            else:
                 time.sleep(5)
                 write_to_log(logfile_path, "status code: " + str(response.status_code) + " " + response.reason)
        
        # Check if there are models in the response
        if 'items' in data:
            # Extract 'id' field from each model and add it to the list
            write_to_log(logfile_path, "totalcnt = " + str(data['metadata'].get('totalItems')))
            write_to_log(logfile_path, "page = " + str(data['metadata'].get('currentPage')) + " of " + str(data['metadata'].get('totalPages')))
            for each in data['items']:
                id = each.get('id')
                name = each.get('name')
                write_to_log(logfile_path, "processing " + str(id) + f" ({name})")

 
                for each1 in each['modelVersions']:
                    model = each1.get('name')
                    model_id = each1.get('id')
                        
                    for file in each1['files']:
                            
                        if file.get('type') =="Training Data":
                            
                            write_to_log(successfile_path, "found training data")
                            
                            downloadurl = file.get('downloadUrl' )
                            downloadmodel = file.get('id' )
                            #downloadurl = f'https://civitai.com/api/download/training-data/:{model_id}'
                            #downloadurl = f'https://civitai.com/api/download/training-data/{model_id}'
                            #downloadurl = f'https://civitai.com/api/download/models/:{downloadmodel}?type=Training%20Data'
                            #downloadfilename = name + '_' + model + '_' + file.get('name')
                            downloadfilename = str(model_id) + '_' + str(file.get('id')) + '_' + file.get('name')
                            download_fullpath = os.path.join(download_to,downloadfilename)  

                            #downloadheader = 'attachment; filename="Tags.zip"'      
                            #headers = {'Content-Type': 'application/json'}

                            if not os.path.exists(download_fullpath):
                                while True:
                                    write_to_log(logfile_path, "downloading " + downloadurl + ".  Filename: " + downloadfilename + ". size: " + str(file.get('sizeKB')))
                                    try:
                                        #downloadurl2 = downloadurl + '?type=Training%20Data'
                                        response = requests.get(downloadurl,headers=headers,allow_redirects=True,stream=True)
                                    except Exception as e:
                                        write_to_log(logfile_path, "Error " + str(e))

                                    if response.status_code == 400:
                                        print("400 file not found")
                                        break
                                    if response.status_code == 404:
                                        print("404 file not found")
                                        break

                                    if response.status_code == 401 and api_key:
                                        # Retry the request with the API key
                                        headers["Authorization"] = f"Bearer {api_key}"
                                        response = requests.get(downloadurl, headers=headers)

                                    if response.status_code == 200:
                                        content_disposition = response.headers.get("Content-Disposition", None)

                                        with open(download_fullpath, 'wb') as file2:
                                            file2.write(response.content)
                                        write_to_log(logfile_path, f"File downloaded successfully to {download_fullpath}")
                                        write_to_log(successfile_path, f"{model},{downloadurl},{downloadfilename}")
                                        break
                                    else:
                                        write_to_log(logfile_path, f"Failed to download file. Status code: {response.status_code}")
                            else:
                                write_to_log(logfile_path, "file already exists")
                        else:
                            write_to_log(logfile_path, "file type is" + f": {file.get('type')}.  Filename: {file.get('name')}")

            # Check if there are more pages
            if data['metadata'].get('currentPage') == (data['metadata'].get('totalPages')):
                print("ran out of pages")
                break 
            else:
                page += 1
        else:
            print("no items found")
            break

download_to = '/folder/to/download/to'


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



logfile_path = os.path.join(download_to,'logfile.log')
successfile_path = os.path.join(download_to,'successfile.log')

get_models()
