import requests
import time
import os
from pathlib import Path
import json

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

def get_paginated_data(url, headers,params=None):
    all_data = []
    cursor = None

    while True:
        # Set the cursor in the request parameters if it exists
        if cursor:
            params['cursor'] = cursor

        # Make the HTTP request
        response = requests.get(url, params=params,headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Process the current page of data
            current_data = response.json()
            all_data.extend(current_data)

            # Check if there's more data to retrieve
            if 'cursor' in current_data:
                cursor = current_data['cursor']
            else:
                break  # No more data

        else:
            print(f"Error: {response.status_code}")
            break  # Stop the loop on error

    return all_data

def cursorpull(url,headers=None,params=None):
    cursor = None

    with requests.Session() as req:
        allin = []
        for _ in range(1, 4):
            r = req.get(url, params=params,headers=headers)
            if r.status_code == 200:

                body = r.json()
                if 'cursor' in r:
                        cursor = r['cursor']
                else:
                    break  # No more data
                
                if cursor:
                    params['cursor'] = cursor
                print(params)

                allin.append(r)
            goal = json.dumps(allin, indent=4)
            with open("data.txt", 'w') as f:
                f.write(goal)


def get_models():

    global api_key
    global UserToDL
    global prompt_file_location
    # Initialize the first page
    page = 1
    batchsize = 100
    cursor = None

    headers = {'Content-Type': 'application/json'}

    req = f'https://civitai.com/api/v1/images?limit={batchsize}&username={UserToDL}'
    ret = cursorpull(req,headers)

    while True:
        all_data = []

        if cursor:
            params['cursor'] = cursor

        # Make API request for the current page

#        if api_key:
#            headers["Authorization"] = f"Bearer {api_key}"

        params = {'page': page}
        print("processing page " + str(page))
        u = 0
        while True:
            try:
                req = f'https://civitai.com/api/v1/images?limit={batchsize}&username={UserToDL}'
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
                current_data = response.json()
                all_data.extend(current_data)

                # Check if there's more data to retrieve
                if 'cursor' in current_data:
                    cursor = current_data['cursor']
                else:
                    break  # No more data

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


        u += 1
        if u == batchsize:
            u = 1
        
        # Check if there are models in the response
        if 'items' in data:
            # Extract 'id' field from each model and add it to the list
            write_to_log(logfile_path, "totalcnt = " + str(data['metadata'].get('totalItems')))
            write_to_log(logfile_path, "page = " + str(data['metadata'].get('currentPage')) + " of " + str(data['metadata'].get('totalPages')))
            for each in data['items']:
                id = each.get('id')
                name = each.get('name')
                write_to_log(logfile_path, "processing " + str(id) + f" ({name})")

                if each is not None and "meta" in each and each["meta"] is not None and "prompt" in each["meta"]:
                    prompt = each["meta"]["prompt"]
                else:
                    prompt = None

                if prompt != None:
                    write_to_log(prompt_file_location, prompt)
 
        # Check if there are more pages
        if data['metadata'].get('currentPage') == (data['metadata'].get('totalPages')):
            print("ran out of pages")
            break 
        else:
            page += 1


download_to = '/folder/to/download/to'
UserToDL = 'username'
prompt_file_location = '/folder/to/download/to.txt'


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

logfile_path = os.path.join(download_to,'logfile.log')
successfile_path = os.path.join(download_to,'successfile.log')

get_models()
