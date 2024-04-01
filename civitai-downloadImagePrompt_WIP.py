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
from PIL import Image
import zlib
import gzip

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

def dump_to_json(filename,data,writestyle='a'):

    if not isinstance(data, dict):
        data = {"data": data}

    while True:
            try:
                with open(filename, writestyle, encoding='utf-8') as json_file:
                    json.dump(data, json_file, indent=2)  # indent for pretty formatting (optional)
                    json_file.write('\n')
                    return
            except Exception as e:
                print(f"write error.  retry {e}")

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


def parse_generation_parameters(x: str):
    """parses generation parameters string, the one you see in text field under the picture in UI:
```
girl with an artist's beret, determined, blue eyes, desert scene, computer monitors, heavy makeup, by Alphonse Mucha and Charlie Bowater, ((eyeshadow)), (coquettish), detailed, intricate
Negative prompt: ugly, fat, obese, chubby, (((deformed))), [blurry], bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), messy drawing
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 965400086, Size: 512x512, Model hash: 45dee52b
```

    returns a dict with field values
    """

    re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
    re_param = re.compile(re_param_code)
    re_imagesize = re.compile(r"^(\d+)x(\d+)$")
    re_hypernet_hash = re.compile(r"\(([0-9a-f]+)\)$")
    if 'Template' in x:
        print("has template")

    res = {}

    prompt = ""
    negative_prompt = ""

    done_with_prompt = False

    *lines, lastline = x.strip().split("\n")
    if len(re_param.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ''

    for line in lines:
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()
        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    res["Prompt"] = prompt
    res["Negative prompt"] = negative_prompt

    for k, v in re_param.findall(lastline):
        try:
            if len(v) > 0:
                if v[0] == '"' and v[-1] == '"':
                    v = unquote(v)

                m = re_imagesize.match(v)
                if m is not None:
                    res[f"{k}-1"] = m.group(1)
                    res[f"{k}-2"] = m.group(2)
                else:
                    res[k] = v
            else:
                print(f"ignoring key {k} as value is empty")
        except Exception as e:
            print(f"Error parsing \"{k}: {v}\" {e}")

    # Missing CLIP skip means it was set to 1 (the default)
    if "Clip skip" not in res:
        res["Clip skip"] = "1"

    hypernet = res.get("Hypernet", None)
    if hypernet is not None:
        res["Prompt"] += f"""<hypernet:{hypernet}:{res.get("Hypernet strength", "1.0")}>"""

    if "Hires resize-1" not in res:
        res["Hires resize-1"] = 0
        res["Hires resize-2"] = 0

    if "Hires sampler" not in res:
        res["Hires sampler"] = "Use same sampler"

    if "Hires checkpoint" not in res:
        res["Hires checkpoint"] = "Use same checkpoint"

    if "Hires prompt" not in res:
        res["Hires prompt"] = ""

    if "Hires negative prompt" not in res:
        res["Hires negative prompt"] = ""

    #restore_old_hires_fix_params(res)

    # Missing RNG means the default was set, which is GPU RNG
    if "RNG" not in res:
        res["RNG"] = "GPU"

    if "Schedule type" not in res:
        res["Schedule type"] = "Automatic"

    if "Schedule max sigma" not in res:
        res["Schedule max sigma"] = 0

    if "Schedule min sigma" not in res:
        res["Schedule min sigma"] = 0

    if "Schedule rho" not in res:
        res["Schedule rho"] = 0

    if "VAE Encoder" not in res:
        res["VAE Encoder"] = "Full"

    if "VAE Decoder" not in res:
        res["VAE Decoder"] = "Full"

    #skip = set(shared.opts.infotext_skip_pasting)
    #res = {k: v for k, v in res.items() if k not in skip}

    return res

def decode_chunked(download_fullpath, response):

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


    #decoded_chunks = b''
    #for chunk in response.iter_content(decode_unicode=False):
    #    decoded_chunks += chunk
    #return decoded_chunks

def decompress_gzip(data):
    return gzip.decompress(data)

def decompress_deflate(data):
    return zlib.decompress(data, -zlib.MAX_WBITS)

def decompress_compress(data):
    return zlib.decompress(data, zlib.MAX_WBITS | 16)

def has_parameters(filepath, extended=False):

    if os.path.exists(filepath) and os.path.isfile(filepath):
        if filepath.endswith(".png"):
            with Image.open(filepath) as img:
                try:
                    parameter = img.info.get("parameters")
                    if parameter is not None:
                        print(filepath + " has metadata.")
                        if extended == True:
                            res = parse_generation_parameters(parameter)
                            if 'Prompt' in res and 'Seed' in res and 'Sampler' in res:
                                #highlevel looks valid
                                return res, True
                            else:
                                print("insufficient parameters to be considered a prompt")
                                return None, False
                        else:
                            return True
                    else:
                        print("PNG with no metadata")
                        if extended == True:
                            return None,False
                        else:
                            return False
                except Exception as e:
                    print("damaged png file")
        else:
            print("non png files don't have parameters")
            if extended == True:
                return None, False
            else:
                return False


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
    global req
    global onlypng
    global mergedprompt
    # Initialize the first page
    page = 1

    headers = {}
    headers['Content-Type'] = 'application/json'
    headers['content-disposition'] = ''
    #headers["Authorization"] = f"Bearer {apikey}"

    sort = "Most Reactions"
    view = "feed"
    #cursor = "1"
    
    #params = {
    #    'limit' : limit,
    #    'cursor': 1
    #    "favourites":"true"
    #    "sort": sort,
    #    "view": view
    #}

    
    totalcnt = 0
    page = 0       
    while True:
        all_data = []
        page += 1
        throttletime = 5
        while True:
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
                #throttletime += 5
                print(f"incremented the throttle by {throttletime} second")
            else:
                 write_to_log(logfile_path, "status code: " + str(response.status_code) + " " + response.reason)

        # Check if there are models in the response
        if 'items' in data and len(data['items']) > 0:
            numitem = len(data['items'])
            for count, each in enumerate(data['items']):
                totalcnt +=1
                count +=1
                print(f"{count}/{numitem} of page {page}.  Total processed: {totalcnt}.")
                #time.sleep(1)
                id = each.get('id')
                name = each.get('name')
                write_to_log(logfile_path, f"processing {id}")
                imagedownloaded= False

                if each is not None and "meta" in each and each["meta"] is not None and "prompt" in each["meta"]:
                    #prompt = each["meta"]["prompt"]
                    each['url'] = replace_width_with_bob(each['url'])
                    postid = each['postId']
                    picid = each['id']
                    filename = f"{UserToDL}_{postid}_{picid}"
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
                                print(f"Sleeping {n} before retry.")
                                time.sleep(n)
                                continue
                            raise
                        except Exception as e:
                            print(f"A different error.  Sleep and keep retrying until done anyway. {e}")
                            time.sleep(5)

                    if 'Transfer-Encoding' in response.headers:
                        print("chunked")
                        if 'PNG' in str(response.content):
                            print('Chunked PNG file')
                            if ext == None:
                                ext = '.png'             
                        elif 'JFIF' in str(response.content):
                            print('Chunked jpg file')
                            if ext == None:
                                ext = '.jpg'
                        elif 'MP4' in str(response.content):
                            print('Chunked MP4 file')
                            if ext == None:
                                ext = '.mp4'
                        elif 'GIF89' in str(response.content):
                            print('Chunked MP4 file')
                            if ext == None:
                                ext = '.mp4'
                        #elif 'image/gif' in response.headers['content-type']:
                        ##    if ext == None:
                        #        ext = '.gif'

                        else:    
                            print("oof")
                            if ext == None:
                                print("double oomph")
                                continue
                            continue

                    download_fullpath = filename + ext
                    download_fullpath = os.path.join(download_to,download_fullpath)

                    if os.path.exists(download_fullpath):
                        print("assume it was successful")
                    else:
                        
                        #do stuff

                        if 'Transfer-Encoding' in response.headers and response.headers['Transfer-Encoding'] == 'chunked':
                            # Decode chunked response
                            decoded_response = decode_chunked(download_fullpath, response)
                        else:
                            decoded_response = response.content
                        
                        # Check if the response is compressed
                        if 'Content-Encoding' in response.headers:
                            content_encoding = response.headers['Content-Encoding'].lower()
                            if content_encoding == 'gzip':
                                # Decompress gzip response
                                decoded_response = decompress_gzip(decoded_response)
                            elif content_encoding == 'deflate':
                                # Decompress deflate response
                                decoded_response = decompress_deflate(decoded_response)
                            elif content_encoding == 'compress':
                                # Decompress compress response
                                decoded_response = decompress_compress(decoded_response)

                    #redo
                        if onlypng == True and response.headers['content-type'] =='image/jpeg':
                            print("not downloading a jpg as PNG is required")
                            continue
                        elif response.headers['content-type'] =='image/jpeg' and get_prompt == True:
                                print("Because it's a jpeg and we want to download it, we'll save the metadata too")
                                json_file = os.path.join(download_to,filename + '.txt')
                                #dump_to_json(prompt_file_location, each["meta"])
                                dump_to_json(json_file, each["meta"],'w')
                                write_to_log(logfile_path, f"prompt found for {each['url']}")                            

                        if (os.path.exists(download_fullpath)) == True:
                            filesize_is = os.path.getsize(download_fullpath)
                            if file_size != filesize_is:
                                write_to_log(logfile_path, f"File {download_fullpath} is {filesize_is} and should be {file_size}.   Failed download.")
                                with open(download_fullpath, 'wb') as file2:
                                    file2.write(response.content)
                            elif (os.path.exists(download_fullpath)) == True and file_size == filesize_is:
                                write_to_log(logfile_path, f"File is correct size ({filesize_is}).")

                                if has_parameters(download_fullpath):
                                    print("Has params")
                                else:
                                    print("has no params dump meta")
                                    json_file = os.path.join(download_to,filename + '.txt')
                                    #dump_to_json(prompt_file_location, each["meta"])
                                    dump_to_json(json_file, each["meta"],'w')
                                    write_to_log(logfile_path, f"prompt found for {each['url']}")     

                                continue
                        else:
                                with open(download_fullpath, 'wb') as file2:
                                    file2.write(response.content)

                                if has_parameters(download_fullpath):
                                    print("Has params")
                                else:
                                    print("has no params dump meta")
                                    json_file = os.path.join(download_to,filename + '.txt')
                                    #dump_to_json(prompt_file_location, each["meta"])
                                    dump_to_json(json_file, each["meta"],'w')
                                    write_to_log(logfile_path, f"prompt found for {each['url']}")                            
                else:
                    write_to_log(logfile_path, f"** no META found for {each['url']} **")

        req = (data['metadata'].get('nextPage'))
        write_to_log(logfile_path,f"req:  {req}")
        
        if req == None:
            print("no more pages")
            break

download_to = '/folder/to/download/to'
UserToDL = 'username'
prompt_file_location = '/folder/to/download/to.txt'
get_prompt=False
get_image=True
onlypng = False
mergedprompt = True
limit = 10

req = f'https://civitai.com/api/v1/images?username={UserToDL}&limit={limit}'

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
