import requests
import time
import os
from pathlib import Path
import json
import time
import re
import platform
from http import HTTPStatus
from requests.exceptions import HTTPError
from urllib.parse import unquote
from tqdm import tqdm

retry_codes = [
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
]

retries = 3

def replace_width_with_bob(url):
    # Use regular expression to replace /width=* with /bob/
    modified_url = re.sub(r'/width=\d+', '/original=true', url)

    return modified_url

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

def extract_jpeg_filename(url):
    # Split the URL to get the filename
    _, filename = os.path.split(url)

    # If the filename contains query parameters, remove them
    filename = filename.split('?')[0]

    return filename

#@sleep(3)
def get_models():

    global apikey
    global UserToDL
    global prompt_file_location
    # Initialize the first page
    page = 1

    limit = 100
    sort = "Most Reactions"
    view = "feed"
    #cursor = "1"
    headers = {}
    headers['Content-Type'] = 'application/json'
    headers['content-disposition'] = ''
    #headers["Authorization"] = f"Bearer {apikey}"

    #params = {
    #    'limit' : limit,
    #    'cursor': 1
    #    "favourites":"true"
    #    "sort": sort,
    #    "view": view
    #}

    onlypng = True
    #postid = 1416958
    #https://civitai.com/api/v1/models?token={apikey}
    #req = f'https://civitai.com/api/v1/images?postId={postid}'
    #req = f'https://civitai.com/api/v1/images?tag={UserToDL}&limit={limit}'
    req = f'https://civitai.com/api/v1/images?username={UserToDL}'
    #req = f'https://civitai.com/api/v1/images?postId={postid}'
    totalcnt = 0
    page = 0       
    while True:
        all_data = []

        page += 1
        #params['page'] = page
        #print("processing page " + str(page))
        throttletime = 5
        j=0
        while True:
            if j != 0:
                print(f"sleeping {throttletime}")
                time.sleep(throttletime)

            try:
                response = requests.get(req, headers=headers)
                #response = requests.get(req, headers=headers,params=params)
                #test = extract_url_cursor(response)
                #response = requests.get(req, headers=headers, params=params)
            except Exception as e:
                 write_to_log(logfile_path, "Error " + str(e))

            if response.status_code == 200:
                current_data = response.json()
                all_data.extend(current_data)
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

            elif response.status_code == 400:
                print(f"bad request {response.text}")
                break

            elif response.status_code == 404:
                print("not found")
                break

            elif response.status_code == 401 and apikey:
                # Retry the request with the API key
                headers["Authorization"] = f"Bearer {apikey}"
                response = requests.get(req, headers=headers)
                #response = requests.get(req, headers=headers, params=params)
            
            elif response.status_code == 429:
                throttletime += 5
                print(f"incremented the throttle by {throttletime} second")
            else:
                 write_to_log(logfile_path, "status code: " + str(response.status_code) + " " + response.reason)
            j += 1

        # Check if there are models in the response
        if 'items' in data and len(data['items']) > 0:
            # Extract 'id' field from each model and add it to the list
            #write_to_log(logfile_path, "totalcnt = " + str(data['metadata'].get('totalItems')))
            #write_to_log(logfile_path, "page = " + str(data['metadata'].get('currentPage')) + " of " + str(data['metadata'].get('totalPages')))

            req = (data['metadata'].get('nextPage'))
            #totes = (data['metadata'].get('pageSize'))
            for count, each in enumerate(data['items']):
                totalcnt +=1
                count +=1
                print(f"{count} of page {page}.  Total processed: {totalcnt}.")
                time.sleep(1)
                id = each.get('id')
                name = each.get('name')
                write_to_log(logfile_path, f"processing {id}")

                if each is not None and "meta" in each and each["meta"] is not None and "prompt" in each["meta"]:
                    #prompt = each["meta"]["prompt"]
                    each['url'] = replace_width_with_bob(each['url'])
                    postid = each['PostId']
                    picid = each['id']
                    filename = f"{UserToDL}_{postid}{picid}"
                    if get_prompt == True:
                        dump_to_json(prompt_file_location, each["meta"])
                        write_to_log(logfile_path, f"prompt found for {each['url']}")

                    if get_image == True:
                        for n in range(retries):
                            try:
                                response = requests.get(each['url'], headers=headers)
                                #response = requests.get(url, headers=headers, params=params)
                                response.raise_for_status()
                                #if response.headers['Content-Type'] =='text/plain':
                                #    print("This shouldn't happen")
                                #    continue
                                break

                            except HTTPError as exc:
                                code = exc.response.status_code
                                
                                if code in retry_codes:
                                    # retry after n seconds
                                    time.sleep(n)
                                    continue

                                raise
                        #filename = response.headers()
                        if 'Content-Disposition' in response.headers:
                            content_disposition = response.headers['Content-Disposition']
                            filename_start = content_disposition.find('filename=') + len('filename=')
                            suggested_filename = content_disposition[filename_start:].strip(' "')

                            # Unquote the filename if it's URL-encoded
                            suggested_filename = unquote(suggested_filename)

                            print(content_disposition)
                        else:
                            if response.headers['content-type'] =='image/jpeg':
                                download_fullpath = extract_jpeg_filename(each['url'])
                            elif response.headers['content-type'] =='image/png':
                                download_fullpath = extract_jpeg_filename(each['url'])
                                download_fullpath = download_fullpath.replace('jpeg','png')

                        download_fullpath = os.path.join(download_to,download_fullpath)
                        #download_fullpath = 'c:/users/simon/desktop/1.png'
                        file_size = int(response.headers.get("content-length", 0))

                        if 'image/jpeg' not in response.headers['content-type'] and 'image/png' not in response.headers['content-type'] and 'Transfer-Encoding' not in response.headers:
                            print(response.headers['content-type'])
                        
                        if 'Transfer-Encoding' in response.headers:
                            print("chunked")
                            if 'PNG' in str(response.content):
                                print('Chunked PNG file')
                            elif 'JFIF' in str(response.content):
                                print('Chunked PNG file')
                            else:    print("oof")
                            if response.headers['Transfer-Encoding'] == 'chunked':
                                try:
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
                                except Exception as e:
                                    print(f"failed {e}")

                        else:
                            print("not chunked")

                            if onlypng == True and response.headers['content-type'] =='image/jpeg':
                                print("not downloading a jpg as PNG is required")
                                continue

                            if (os.path.exists(download_fullpath)) == True:
                                filesize_is = os.path.getsize(download_fullpath)
                                if file_size != filesize_is:
                                    write_to_log(logfile_path, f"File {download_fullpath} is {filesize_is} and should be {file_size}.   Failed download.")
                                    with open(download_fullpath, 'wb') as file2:
                                        file2.write(response.content)
                                elif (os.path.exists(download_fullpath)) == True and file_size == filesize_is:
                                    write_to_log(logfile_path, f"File is correct size ({filesize_is}).  Next..")
                                    continue
                            else:
                                    with open(download_fullpath, 'wb') as file2:
                                        file2.write(response.content)
                else:
                    write_to_log(logfile_path, f"no prompt found for {each['url']}")

        # Check if there are more pages
        #if (data['metadata'].get('totalPages')) == None:
        #    print("no such section.  just increment the page number")

        #if data['metadata'].get('currentPage') == (data['metadata'].get('totalPages')):
        #    print("ran out of pages")
        #    break
        #el
        
        if req == None:
            print("no more pages")
            break


download_to = '/folder/to/download/to'
UserToDL = 'username'
prompt_file_location = '/folder/to/download/to.txt'
get_prompt=True
get_image=True

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
