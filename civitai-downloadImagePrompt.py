import requests
import time
import os
from pathlib import Path
import json
import time
import re
import platform

def dump_to_json(filename,data):

    if not isinstance(data, dict):
        data = {"data": data}

    with open(filename, 'a', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=2)  # indent for pretty formatting (optional)
        json_file.write('\n')

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

def write_to_log(log_file, message):
    print(message)
    try:
        with open(log_file, 'a', encoding='utf-8') as file:
            file.write(message + '\n')
    except Exception as e:
        print(f"Error writing to the log file: {e}")

def sleep(timeout, retry=3):
    def the_real_decorator(function):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < retry:
                try:
                    value = function(*args, **kwargs)
                    if value is None:
                        return
                except:
                    print(f'Sleeping for {timeout} seconds')
                    time.sleep(timeout)
                    retries += 1
        return wrapper
    return the_real_decorator

def extract_url_cursor(url):
    model_cursor_match = re.search(r'cursor=(\d+)', url)

#@sleep(3)
def get_models():

    global apikey
    global UserToDL
    global prompt_file_location
    # Initialize the first page
    page = 1
    batchsize = 10

    limit = 10
    sort = "Most Reactions"
    view = "feed"
    #cursor = "1"
    headers = {}
    headers['Content-Type'] = 'application/json'
    #headers["Authorization"] = f"Bearer {apikey}"

    params = {
        'limit' : limit,
    #    'cursor': 1
    #    "favourites":"true"
    #    "sort": sort,
    #    "view": view
    }
    while True:
        all_data = []

        #params['page'] = page
        #print("processing page " + str(page))
        u = 0
        throttletime = 5
        j=0
        while True:
            if j != 0:
                print(f"sleeping {throttletime}")
                time.sleep(throttletime)

            try:
                #https://civitai.com/api/v1/models?token={apikey}
                req = f'https://civitai.com/api/v1/images?tag={UserToDL}'
                #req = f'https://civitai.com/api/v1/images?username={UserToDL}&page={page}'
                
                response = requests.get(req, headers=headers,params=params)
                #test = extract_url_cursor(response)
                #response = requests.get(req, headers=headers, params=params)
            except Exception as e:
                 write_to_log(logfile_path, "Error " + str(e))

            if response.status_code == 200:
                current_data = response.json()
                all_data.extend(current_data)

                # Check if there's more data to retrieve
                if 'cursor' in current_data:
                    cursor = current_data['cursor']
                #else:
                #    break  # No more data

                try:
                    data = response.json()
                    if 'items' in data:
                        print("found items.  Legitimate data")
                        break
                except Exception as e:
                    write_to_log(logfile_path, "Error " + str(e))

            elif "be back in a few minutes" in str(response.content):
                print("error.  Site down")
                exit()

            elif response.status_code == 404:
                print("not found")
                break

            elif response.status_code == 401 and apikey:
                # Retry the request with the API key
                headers["Authorization"] = f"Bearer {apikey}"
                response = requests.get(req, headers=headers, params=params)
            
            elif response.status_code == 429:
                throttletime += 5
                print(f"incremented the throttle by {throttletime} second")
            else:
                 write_to_log(logfile_path, "status code: " + str(response.status_code) + " " + response.reason)
            j += 1

        u += 1
        if u == batchsize:
            u = 1
        
        # Check if there are models in the response
        if 'items' in data and len(data['items']) > 0:
            # Extract 'id' field from each model and add it to the list
            write_to_log(logfile_path, "totalcnt = " + str(data['metadata'].get('totalItems')))
            write_to_log(logfile_path, "page = " + str(data['metadata'].get('currentPage')) + " of " + str(data['metadata'].get('totalPages')))

            totes = (data['metadata'].get('pageSize'))
            r = 0
            for each in data['items']:
                r += 1
                print(f"processing {r}/{totes} page {data['metadata'].get('currentPage')}/{data['metadata'].get('totalPages')}")
                time.sleep(1)
                id = each.get('id')
                name = each.get('name')
                write_to_log(logfile_path, f"processing {id}")

                if each is not None and "meta" in each and each["meta"] is not None and "prompt" in each["meta"]:
                    #prompt = each["meta"]["prompt"]
                    dump_to_json(prompt_file_location, each["meta"])
                    write_to_log(logfile_path, f"prompt found for {each['url']}")
                else:
                    prompt = None
                    write_to_log(logfile_path, f"no prompt found for {each['url']}")

        # Check if there are more pages
        if (data['metadata'].get('totalPages')) == None:
            print("no such section.  just increment the page number")

        elif data['metadata'].get('currentPage') == (data['metadata'].get('totalPages')):
            print("ran out of pages")
            break

        page += 1


download_to = '/folder/to/download/to'
UserToDL = 'username'
prompt_file_location = '/folder/to/download/to.txt'


apifile = os.path.join(get_script_path(), "apikey.py")
if os.path.exists(apifile):
    exec(open(apifile).read())
    apikey = apikey
    print("API Key:", apikey)
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
    #apikey = apikey
    #print("API Key:", apikey)
    print("local override file is " + localoverridesfile)

else:
    print("local override file would be " + localoverridesfile)



logfile_path = os.path.join(download_to,'logfile.log')
successfile_path = os.path.join(download_to,'successfile.log')

get_models()
