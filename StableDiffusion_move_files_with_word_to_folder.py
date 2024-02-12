import os
import shutil
from PIL import Image
from pathlib import Path
import platform
import json
import re
import hashlib
import blake3
import binascii
import zlib
    
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

def unquote(text):
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text

    try:
        return json.loads(text)
    except Exception:
        return text
    
def parse_generation_parameters(x: str):
    """parses generation parameters string, the one you see in text field under the picture in UI:
```
girl with an artist's beret, determined, blue eyes, desert scene, computer monitors, heavy makeup, by Alphonse Mucha and Charlie Bowater, ((eyeshadow)), (coquettish), detailed, intricate
Negative prompt: ugly, fat, obese, chubby, (((deformed))), [blurry], bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), messy drawing
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 965400086, Size: 512x512, Model hash: 45dee52b
```

    returns a dict with field values
    """

    re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|>$)'
    re_param = re.compile(re_param_code)
    re_imagesize = re.compile(r"^(\d+)x(\d+)$")
    re_hypernet_hash = re.compile("\(([0-9a-f]+)\)$")

    index_of_negative = x.find('\nNegative')
    if index_of_negative != -1:
        # Replace '\n' with an empty string only before '\nNegative'
        y = x
        x = x[:index_of_negative].replace('\n', '') + x[index_of_negative:]

    index_of_negative = x.find('\nnegative')
    if index_of_negative != -1:
        # Replace '\n' with an empty string only before '\nNegative'
        y = x
        x = x[:index_of_negative].replace('\n', '') + x[index_of_negative:]

    res = {}

    prompt = ""
    negative_prompt = ""

    done_with_prompt = False

    if 'cfg scale:' in x:
        x = x.replace('cfg scale:','CFG scale:')

    *lines, lastline = x.strip().split("\n")

    if 'steps:' not in lastline[6]:
        if 'negative prompt:' in x or 'Negative prompt:' in x:
            totalnumlines = 2
        else:
            totalnumlines = 1

        while len(lines) > totalnumlines:
            i = totalnumlines
            while i < len(lines):
                print(lines[i])
                if 'seed:' in lines[i]:
                    lastline = lines[i]
                lines.pop(i)
            i += 1

    lastlinefindall = re_param.findall(lastline)
    if len(lastlinefindall) < 3:
        lines.append(lastline)
        lastline = ''

    for line in lines:
        line = line.strip()
        if line.startswith("Negative prompt:") or line.startswith("negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()
        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    if 'model: ' in negative_prompt.lower() or 'model: ' in prompt.lower():
        print(f"Model: should not be in a prompt.\nNegative: {negative_prompt}\nPositive: {prompt}")

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
                    if k != 'CFG scale':
                        res[k.title()] = v
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

def sanitise_path_name(folder_name):
    # Define a regular expression pattern to match invalid characters
    invalid_chars_pattern = re.compile(r'[\\/:"*?<>| ]')

    # Replace invalid characters with an empty string
    sanitised_folder_name = re.sub(invalid_chars_pattern, '', folder_name)

    return sanitised_folder_name

def generate_file_hashes(file_path):
    # Initialize hash objects
    sha256_hash = hashlib.sha256()
    md5_hash = hashlib.md5()
    blake3_hasher = blake3.blake3()
    blake2_hash = hashlib.blake2b(digest_size=32)
    crc32_hash = 0  # Initialize CRC32

    # Process the file in chunks for efficiency
    chunk_size = 8192  # You can adjust this based on your needs

    with open(file_path, "rb") as file:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        file.seek(0)
        header = file.read(8)
        n = int.from_bytes(header, "little")

        offset = n + 8
        file.seek(offset)
        for chunk in iter(lambda: file.read(blksize), b""):
            hash_sha256.update(chunk)
        
        addnet8 = hash_sha256.hexdigest()[0:8]
        addnet12 = hash_sha256.hexdigest()[0:12]

    with open(file_path, "rb") as file:
        m = hashlib.sha256()
        file.seek(0x100000)
        m.update(file.read(0x10000))
        model_hash =  m.hexdigest()[0:8]
    
    with open(file_path, 'rb') as file:
        while chunk := file.read(chunk_size):
            sha256_hash.update(chunk)
            md5_hash.update(chunk)
            blake3_hasher.update(chunk)
            blake2_hash.update(chunk)
            crc32_hash = zlib.crc32(chunk, crc32_hash)

    # Get the hexadecimal representations of the hashes
    sha256_hex = sha256_hash.hexdigest()
    md5_hex = md5_hash.hexdigest()
    blake3_hex = blake3_hasher.hexdigest()
    blake2_hex = blake2_hash.hexdigest()
    crc32_hex = format(crc32_hash & 0xFFFFFFFF, '08x')

    return {
        'SHA256': sha256_hex,
        'MD5': md5_hex,
        'BLAKE3': blake3_hex,
        'BLAKE2': blake2_hex,
        'CRC32': crc32_hex,
        'model_hash' : model_hash,
        'add_net8' : addnet8,
        'add_net12' : addnet12,
    }


def extract_json_from_file(file_path, max_bytes=20 * 1024 * 1024):
    # Read the first specified number of bytes from the file in binary mode
    #with open(file_path, 'rb') as file:
    #    binary_data = file.read(max_bytes)
    
        #b8=file.read(8) #read header size
    print(f"extracting JSON from {file_path}")

    val = read_safetensor_header(file_path)
    try:
        json_data = json.loads(val)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except UnicodeDecodeError as e:
        # Handle the error, print the position and the problematic character
        print(f"Error decoding file: {e}")
        print(f"Problematic character at position {e.start}.  {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

    if '__metadata__' in json_data:
        result = json_data['__metadata__']

        retarray = {}
        if 'ss_new_sd_model_hash' in result:
            retarray['ss_new_sd_model_hash'] = result['ss_new_sd_model_hash']
        if 'sshs_model_hash' in result:
            retarray['sshs_model_hash'] = result['sshs_model_hash']

        if 'sshs_legacy_hash' in result:
            retarray['sshs_legacy_hash'] = result['sshs_legacy_hash']
        if 'ss_sd_model_hash' in result:
            retarray['ss_sd_model_hash'] = result['ss_sd_model_hash']
        if 'ss_sd_model_hash' in result:
            retarray['ss_sd_model_hash'] = result['ss_sd_model_hash']
        if 'ss_sd_scripts_commit_hash' in result:
            retarray['ss_sd_scripts_commit_hash'] = result['ss_sd_scripts_commit_hash']
        if 'ss_new_vae_hash' in result:
            retarray['ss_new_vae_hash'] = result['ss_new_vae_hash']
        if 'ss_vae_hash' in result:
            retarray['ss_vae_hash'] = result['ss_vae_hash']

        #for each in json_data:
        #    print(each)

        return retarray
    else:
        return None

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
    return None,None  # Return None if the term is not found in any entry

def find_filepath_by_hash(hash_value, data):
    if hash_value in str(data):
        hashlen = len(hash_value)
        for filepath, hashes in data.items():
            #print(f"{filepath}{hashes}{hash_value}")
            for key, value in hashes.items():
                if value is not None:
                    left_substring = value[:hashlen]
                    if hash_value == left_substring:
                        print(f"found hash {hash_value} in {key}. Lora is {hashes['name']}.  trained words are {hashes['trainedWords']} ")
                        return hashes['name']
    return None
def getloras(parameter):

    matches = re.findall(r'<lora:(.*?):', parameter)
    loras = '_'.join(set(matches))

    #loras = '_'.join(matches)

    if len(loras) >0:
        #print("Lora !")
        return loras
    else:
        return None
def read_safetensor_header(file_path):
    print(f"processing {file_path}")
    with open(file_path, 'rb') as file:
        consecutive_zeros_count = 0
        current_byte = b''

        while consecutive_zeros_count < 5:
            current_byte = file.read(1)

            if not current_byte:
                # End of file reached before finding five consecutive zeros
                break

            if current_byte == b'\x00':
                consecutive_zeros_count += 1
            else:
                consecutive_zeros_count = 0

        if consecutive_zeros_count == 5:
            # Continue reading until '00 58' is found
            retval = b''
            while True:
                current_byte = file.read(1)

                if not current_byte:
                    # End of file reached before finding ']]'
                    break

                retval += current_byte

                if current_byte == b']':
                    next_byte_1 = file.read(1)
                    next_byte_2 = file.read(1)

                    if next_byte_1 + next_byte_2 == b'}}':
                        retval += next_byte_1 + next_byte_2
                        #print(retval.decode('utf-8'))
                        return retval
                    else:
                        retval += next_byte_1 + next_byte_2
                        current_byte = next_byte_2

    return None

def process_file(fullpath):
    res = {}

    base, ext = os.path.splitext(fullpath)

    if '.safetensors' in fullpath and '.safetensors.json' not in fullpath and '.safetensors.txt' not in fullpath:
        print(f"Processing {fullpath}")
        saferet = extract_json_from_file(fullpath)
        if saferet != None:
            print(f"{saferet}")
            if base not in res:
                res[base] = {}

            res[base].update({
                'sshs_model_hash': saferet.get('sshs_model_hash', None),
                'sshs_legacy_hash': saferet.get('sshs_legacy_hash', None),
                'ss_new_sd_model_hash': saferet.get('ss_new_sd_model_hash', None),
                'ss_sd_model_hash': saferet.get('ss_sd_model_hash', None),
                'ss_sd_scripts_commit_hash': saferet.get('ss_sd_scripts_commit_hash', None),
                'ss_new_vae_hash': saferet.get('ss_new_vae_hash', None),
                'ss_vae_hash': saferet.get('ss_vae_hash', None),
                })
            
        morehashes = generate_file_hashes(fullpath)
        if morehashes != None:
                print(f"{morehashes}")
                if base not in res:
                    res[base] = {}

                res[base].update({
                    'sha256_hex': morehashes.get('SHA256', None),
                    'md5_hex': morehashes.get('MD5', None),
                    'blake3_hex': morehashes.get('BLAKE3', None),
                    'blake2_hex': morehashes.get('BLAKE2', None),
                    'crc32_hex': morehashes.get('CRC32', None),
                    'model_hash': morehashes.get('model_hash', None),
                    'addnet12': morehashes.get('add_net12', None),
                    'addnet8': morehashes.get('add_net8', None),
                    })

    if '.civitai.info' in fullpath:
        print(f"Processing {fullpath}")
        with open(fullpath, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        base, ext = os.path.splitext(base)

        if '.civitai.info' in fullpath:
            test = json_data['files']
            for each in test:
                if 'hashes' in each:
                    hashes = each['hashes']
                    if base not in res:
                        res[base] = {}

                    res[base].update({
                            'name': json_data.get('model',None).get('name',None),
                            'trainedWords': json_data.get("trainedWords", None),
                            'AutoV2': hashes.get('AutoV2', None),
                            'SHA256': hashes.get('SHA256', None),
                            'CRC32': hashes.get('CRC32', None),
                            'BLAKE3': hashes.get('BLAKE3', None)
                        })

                    print(str(hashes))

    return res


def getlorahashes(foldertosearch):
    result = {}

    if os.path.isfile(foldertosearch):
        print("It's a file")
        result = process_file(foldertosearch)

    elif os.path.isdir(foldertosearch):
        print("It's a directory")
        for root, dirs, files in os.walk(foldertosearch):
            for filename in files:
                fullfilepath = os.path.join(root,filename)
                result.update(process_file(fullfilepath))

    else:
        print("oops")
 
    print("Hashing complete")
    return result

def get_imagemeta(file_path):
    try:
        with Image.open(file_path) as img:
            parameter = img.info.get("parameters")
            if parameter is not None:
                print(file_path + " has metadata.")
                return parameter
            else:
                print("PNG with no metadata")
                return None
    except Exception as e:
        print(f"error {e}")
        return None

def search_and_move_files(search_term_array,foldertoSearch):
    global lorafolder
    global embeddedfolder
    global cache
    global move_file

    if os.path.exists(cache):
        with open(cache, 'r') as json_file:
            res1 = json.load(json_file)
            #for each in res1:
            #    print(each)
                #res1.update(getlorahashes("x:/dif/stable-diffusion-webui-docker/data/models/Lora/Clothing/8682_10240_bunnysuit.safetensors"))
    else:
        res1 = {}
        res1.update(getlorahashes(embeddedfolder))
        res1.update(getlorahashes(lorafolder))
        # Dump the dictionary to the JSON file
        with open(cache, 'w') as json_file:
            json.dump(res1, json_file, indent=2)  # The 'indent' parameter is optional for pretty formatting

    #search_term_array = [term for entry in search_term_array for term in entry["terms"]]
    #search_term_array_lower = [term.lower() for entry in search_term_array for term in entry["terms"]]

#    search_term_array_lower = [
#    {"terms": [term.lower() for term in entry["terms"]]} 
#    for entry in search_term_array
#    ]

#    for each in search_term_array:
#        print(each)
#        print("test")
#        for term in each['terms']:
#            term = term.lower()

    for search_term in search_term_array:
        search_term['terms'] = [term.lower() for term in search_term['terms']]

    print("searching " + foldertoSearch)

    for root, dirs, files in os.walk(foldertoSearch):
        for file in files:
            print("processing " + file)
            file_path = os.path.join(root, file)
            if not os.path.isfile(file_path):
                continue
            hasparameters = False
            found = False
            checkfilename = True

            for search_term in search_term_array:
                #print(search_term)

                for term in search_term['terms']:
                    if term.lower() in file.lower():
                        print(f'search term {term.lower()} exists in filename {file.lower()}')

                        keyword, foldername = find_folder_for_term(search_term_array,term.lower())

                        if keyword != None:
                            movetofixedfolder = True
                        else:
                            movetofixedfolder = False

                        if checkfilename == True:
                            if move_file == True:
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
                    parameter = get_imagemeta(file_path)
                    if parameter is not None:
                        print(file_path + " has metadata.")
                        hasparameters = True
                        parameter = parameter.lower()
                    else:
                        print("PNG with no metadata")
                        badfile = True
                if hasparameters ==True:
                    #if any(terms) in parameter:
                    result = parse_generation_parameters(parameter)
                    res = None
                    aloras = ""
                    if 'Lora Hashes' in result:
                        #print("lora hashes in parsed")
                        loras = (result['Lora Hashes']).split(', ')
                        loras = list(set(loras))
                        for lora in loras:
                            deets = lora.split(': ')
                            loraname = deets[0]
                            lorahash = deets[1]
                            res = find_filepath_by_hash(lorahash,res1)
                            if res != None:
                                #print(f"{res}")
                                aloras = f"{aloras}{res}_{lorahash}_"
                            else:
                                print(f"no hash found for {loraname}")
                    elif 'lora' in result['Prompt']:
                        print("it's in prompt")
                        aloras = getloras(result['Prompt'])
                    else:
                        print("no loras")
                    if renamefiles == True:
                        platform = "A1111"
                        seed = 'Seed_' + result.get('Seed', "None")
                        model = 'model_' + result.get('Model', "None")
                        new_filename = f"{platform}_{model}_{seed}"

                        if aloras != "":
                            new_filename = f"{new_filename}_loras_{aloras}"

                        new_filename = new_filename + os.path.splitext(file)[1]
                        new_filename = sanitise_path_name(new_filename)
                        new_item_path = os.path.join(root, new_filename)

                        print(new_item_path)

                        if file_path not in new_item_path:

                            # Handle duplicate filenames
                            base, ext = os.path.splitext(new_item_path)
                            count = 1
                            while os.path.exists(new_item_path):
                                new_item_path = f"{base}_{count}{ext}"
                                count += 1

                            if os.path.abspath(file_path) == os.path.abspath(new_item_path):
                                print("Source and destination are the same. No move needed.")
                                return

                            try:
                                if os.path.exists(new_item_path):
                                    print("This should never occur")
                                else:
                                    new_item_path = os.path.normpath(new_item_path)
                                    shutil.move(file_path, new_item_path)
                                    file_path = new_item_path
                            except Exception as e:
                                print(f"Error moving '{file_path}' to '{new_item_path}': {str(e)}")

                        else:
                            print("doesn't need renaming.  Src and dest are the same: " + file_path + ' ' + new_item_path)
                    for search_term in search_term_array:
                        for term in search_term['terms']:
                            if term.lower() in parameter.lower():
                                print(f"Found '{term.lower()}' in parameters for : {file_path}")

                                if term =='gwenten':
                                    print("check")
                                keyword, foldername = find_folder_for_term(search_term_array,term.lower())

                                if keyword != None:
                                    movetofixedfolder = True
                                else:
                                    movetofixedfolder = False

                                #user_input = input("Do you want to move this file? (y/n): ").strip().lower()
                                #if user_input == 'y':
                                if move_file == True:
                                    if movetofixedfolder == True and keyword != None and foldername !=None:
                                        move_file_to_fixedfolder(file_path ,foldername,keyword)
                                    elif movetofixedfolder == False and foldername !=None:
                                        move_file_to_subfolder(file_path, foldername)
                                    else:
                                        print("Could not move file")

                            #else:
                            #    print(search_term + " not found in parameters for " + file_path)

                else:
                    print(f"no parameters.  Skipping {file_path}")
                    continue



# Directory to search
search_directory =  '/path/to/search'
destination =  '/folder/to/move/to'
lorafolder = '/folder/with/loras'
embeddedfolder = '/folder/with/embedded'
cache = ''
renamefiles = False
move_file = False
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
