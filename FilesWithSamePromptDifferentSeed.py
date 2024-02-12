import os
import re
import hashlib
from datetime import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo
#import uuid
import csv
from collections import defaultdict
from collections import Counter
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import platform
import json

#python3.11 -m venv venv
#source ./venv/bin/activate
#pip install nltk
#pip install pillow
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def get_script_name():
    # Use os.path.basename to get the base name (script name) from the full path
    #basename = os.path.basename(path)
    return Path(__file__).stem
    #return os.path.basename(__file__)

def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))

def get_operating_system():
    system = platform.system()
    return system

def write_to_log(log_file, message):
    global debug
    if debug == True: print(message)
    try:
        with open(log_file, 'a', encoding='utf-8') as file:
            file.write(message + '\n')
    except Exception as e:
        print(f"Error writing to the log file: {e}.  Press a key to continue")
        input()

def find_highest_numbered_folder(directory):
    try:
        # List all directories in the specified path
        subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

        numeric_parts = [int(''.join(filter(str.isdigit, d))) for d in subdirectories if d.isdigit()]
        #numeric_parts = [d for d in subdirectories if d.isdigit()]
        # Extract numeric parts and convert to integers
        #numeric_parts = [int(''.join(filter(str.isdigit, d))) for d in subdirectories]

        if numeric_parts:
            # Find the highest number
            highest_number = max(numeric_parts)
            return highest_number
        else:
            return 0  # No numbered folders found

    except OSError as e:
        print(f"Error: {e}")
        return None


def read_style_to_list(file_path):
    data_array = []

    #with open(file_path, 'r', encoding='utf-8') as f:
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        
        try:
            reader = csv.DictReader(f, delimiter=',')

            for row in reader:
                #row_lower = {key: value.lower() for key, value in row.items()}
                #row_lower = {key: value.lower() if value else '' for key, value in row.items()}

                name = row['name'].lower()
                pos_prompt = row['prompt'].lower()
                neg_prompt = row['negative_prompt'].lower()

                if pos_prompt == '' and neg_prompt == '':
                    print("ignoring delimiter " + name + " " + str(row))
                    continue
                else:
                    #data_array.append(row_lower)

                    data_array.append({
                        'name': name,
                        'prompt': pos_prompt,
                        'negative_prompt': neg_prompt
                    })

        except Exception as e:
            print("Error with style file " + file_path + " " + str(e))


    return data_array

def create_word_groups_parallel_func(args):
    filepath_strings, start, end = args
    result_filepaths = []

    # Convert the dictionary items to a list for easy iteration
    items = list(filepath_strings.items())[start:end]

    stop_words = set(stopwords.words('english'))

    for i in range(len(items)):
        current_filepath, current_string = items[i]

        current_group = [current_filepath]
        current_words = set(word_tokenize(current_string.lower())) - stop_words

        for j in range(i + 1, len(items)):
            compare_filepath, compare_string = items[j]
            compare_words = set(word_tokenize(compare_string.lower())) - stop_words

            # Calculate the percentage of common words
            percentage_common = len(current_words.intersection(compare_words)) / max(len(current_words), len(compare_words)) * 100

            # If over 90% of words are the same, add the filepath to the current group
            if percentage_common > 90:
                current_group.append(compare_filepath)

        # Add the current group to the result if it contains more than one filepath
        if len(current_group) > 1:
            result_filepaths.append(current_group)

    return result_filepaths

def create_word_groupsbywords(filepath_strings, max_different_words=1, group_if_over=1):
    word_groups = []

    # Convert the dictionary items to a list for easy iteration
    items = list(filepath_strings.items())

    while items:
        current_filepath, current_string = items.pop(0)

        current_group = [current_filepath]
        current_words = set(word_tokenize(current_string[0].lower()))

        i = 0
        while i < len(items):
            print(str(i) + " of " + str(len(items)))
            compare_filepath, compare_string = items[i]
            compare_words = set(word_tokenize(compare_string[0].lower()))

            # Remove stop words (optional)
            stop_words = set(stopwords.words('english'))
            current_words = current_words - stop_words
            compare_words = compare_words - stop_words

            # Calculate the number of different words
            different_words = len(current_words.symmetric_difference(compare_words))

            # If the number of different words is less than or equal to the specified value, add the filepath to the current group
            if different_words <= max_different_words:
                current_group.append(compare_filepath)
                items.pop(i)  # Remove the compared item from the list
            else:
                i += 1

        # Add the current group to the list if it contains more than one filepath
        if len(current_group) > group_if_over:
            word_groups.append(current_group)

    return word_groups

def create_word_groups_precentage(filepath_strings,percentagesimilar=90,groupifover=1):
    word_groups = []

    # Convert the dictionary items to a list for easy iteration
    items = list(filepath_strings.items())

    while items:
        current_filepath, current_string = items.pop(0)

        current_group = [current_filepath]
        current_words = set(word_tokenize(current_string[0].lower()))

        i = 0
        while i < len(items):
            print(str(i) + " of " + str(len(items)))
            compare_filepath, compare_string = items[i]
            compare_words = set(word_tokenize(compare_string[0].lower()))

            # Remove stop words (optional)
            stop_words = set(stopwords.words('english'))
            current_words = current_words - stop_words
            compare_words = compare_words - stop_words

            # Calculate the intersection of the sets
            common_words = current_words.intersection(compare_words)

            # Calculate the percentage of common words
            percentage_common = len(common_words) / max(len(current_words), len(compare_words)) * 100

            # If over 90% of words are the same, add the filepath to the current group
            if percentage_common > percentagesimilar:
                current_group.append(compare_filepath)
                items.pop(i)  # Remove the compared item from the list
            else:
                i += 1

        # Add the current group to the list if it contains more than one filepath
        if len(current_group) > groupifover:
            word_groups.append(current_group)

    return word_groups

def create_word_groups_parallel(filepath_strings, num_processes=4):
    word_groups = []
    items = list(filepath_strings.items())

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        chunk_size = len(items) // num_processes
        futures = [executor.submit(create_word_groups_parallel_func, (filepath_strings, i, i + chunk_size)) for i in range(0, len(items), chunk_size)]

        for future in futures:
            word_groups.extend(future.result())

    return word_groups

def checkposandnegarrays(data_dict, positivepromptarray, negativepromptarray):

    posprompt = False
    negprompt = False

    pos1 = None
    neg1 = None
    cntr = 0

    #print(type(data_dict))
    if positivepromptarray != None :
        #print("Pos array is not empty.  Len is " + str(len(positivepromptarray)))
        posprompt = True

    if negativepromptarray != None :
        #print("neg array is not empty.  Len is " + str(len(negativepromptarray)))
        negprompt = True

    for entry in data_dict:
       # print(entry)
        #prompt = False
        styleposprompt = entry['prompt']
        stylenegprompt = entry['negative_prompt']
        stylename = entry['name']

        #print("style name: " + str(stylename))
        #print("style positive prompt: " + str(styleposprompt))
        #print("style negative prompt: " + str(stylenegprompt))
        #print("check against positive of " + str(positivepromptarray))
        #print("check against negative of " + str(negativepromptarray))
        pos = False
        neg = False
        #if 'latex' in stylename:
        #    print("test")
        if str(styleposprompt) != '' and posprompt == True:
            if '{prompt}' in styleposprompt:
                #print('{prompt} present in positive prompt: ' + styleposprompt)
                styleposprompt = styleposprompt.replace('{prompt}','')
                #del my_dict['key2']
                #print('{prompt} removed from positive prompt: ' + styleposprompt)

            styleposprompt = styleposprompt.replace(', ',',')
            styleposprompt = styleposprompt.replace(',,',',')
            styleposprompt = styleposprompt.split(',')
            styleposprompt = sorted(styleposprompt)

            #this is for list
            #stylenegprompt = {key: value for key, value in stylenegprompt.items() if value}
            
            #this i for disct
            styleposprompt = [item for item in styleposprompt if item]

            #pos = all(item in styleposprompt for item in positivepromptarray)
            pos = all(item in positivepromptarray for item in styleposprompt)
            posopp = all(item not in styleposprompt for item in positivepromptarray)
    #    else:
    #        print("Style has empty positive prompt or prompt is empty.  will always match")

        if str(stylenegprompt) != '' and negprompt == True:
            if '{prompt}' in stylenegprompt:
                #print('{prompt} present in negative prompt: '  + stylenegprompt)
                stylenegprompt = stylenegprompt.replace('{prompt}','')               
                #print('{prompt} removed from negative prompt: ' + stylenegprompt)
 
            stylenegprompt = stylenegprompt.replace(', ',',')
            stylenegprompt = stylenegprompt.replace(',,',',')
            stylenegprompt = stylenegprompt.split(',')
            stylenegprompt = sorted(stylenegprompt)

            #this is for list
            #stylenegprompt = {key: value for key, value in stylenegprompt.items() if value}
            
            #this i for dict
            stylenegprompt = [item for item in stylenegprompt if item]
            neg = all(item in negativepromptarray for item in stylenegprompt)
            negopp = all(item not in stylenegprompt for item in negativepromptarray)

    #    else:
    #        print("Style has Empty negative prompt or negprompt is empty.  Will always be a match")

        if (pos or styleposprompt == '' ) and (neg or stylenegprompt == ''):
            #print("Neg and positive match for a style")
            cntr += 1
            if pos:
                #print("positive matches")
                promp = entry['name']
                #print("All members of array2 are present in array1")
                pos1 = [item for item in positivepromptarray  if item not in styleposprompt]
                # Add a single value 'match' to array1
                pos1.append('z_' + promp + '__')
                #print('replaced ' + str(positivepromptarray) + " with " + str(pos1))
                positivepromptarray = pos1

            if neg:
                #print("neg matches")
                promp = entry['name']
                #print("All members of array2 are present in array1")
                neg1 = [item for item in negativepromptarray if item not in stylenegprompt ]
                # Add a single value 'match' to array1
                neg1.append('z_' + promp + '__')
                #print('replaced ' + str(negativepromptarray) + " with " + str(neg1))
                negativepromptarray = neg1


    #print("Counters are both,pos,neg" ,bothcnt,poscnt,negcnt )  
    if pos1 ==False and neg1 == False:
        print("no Styles found in prompt")  
    else:
        print("number of matches is " + str(cntr))
        
    return pos1, neg1

def extract_text_after2(list_obj, text):
    for element in list_obj:
        if element.strip().lower().startswith(text.lower() + ":"):
            return element.split(":")[-1].strip()
    return None

def extract_between_angle_brackets(input_string):
    pattern = re.compile(r'<(.*?)>')
    matches = pattern.findall(input_string)
    return matches

def ensure_comma_before_lt(input_string):
    # Use a regular expression to match '<' not preceded by a comma
    pattern = re.compile(r'(?<!,)(<)')
   #pattern = re.compile(r'(?<!,\s*)(<)')
 
    # Insert a comma before '<' using the sub method
    result_string = pattern.sub(r',\1', input_string)
    
    return result_string

def remove_spaces_around_comma(input_string):
    # Use a regular expression to match spaces around commas
    pattern = re.compile(r'\s*,\s*')
    
    # Replace spaces around commas with a comma
    result_string = pattern.sub(',', input_string)
    
    return result_string

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

    re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
    re_param = re.compile(re_param_code)
    re_imagesize = re.compile(r"^(\d+)x(\d+)$")
    re_hypernet_hash = re.compile("\(([0-9a-f]+)\)$")
    if 'Template' in x:
        print("check")

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

def sanitise_path_name(folder_name):
    # Define a regular expression pattern to match invalid characters
    invalid_chars_pattern = re.compile(r'[\\/:"*?<>| ]')

    # Replace invalid characters with an empty string
    sanitised_folder_name = re.sub(invalid_chars_pattern, '', folder_name)

    return sanitised_folder_name

def get_sanitised_download_time(filepath):
    # Get the modification time of the file
    mtime = os.path.getmtime(filepath)
    # Convert the modification time to a datetime object
    dt = datetime.fromtimestamp(mtime)
    # Format the datetime as a string in the format YYYY-MM-DD_HH-MM-SS
    dt_string = dt.strftime("%Y-%m-%d_%H-%M-%S")
    # Replace any spaces or colons with underscores
    sanitised_dt_string = dt_string.replace(" ", "_").replace(":", "_")
    # Return the sanitised datetime string
    return sanitised_dt_string

def getloras(parameter):

    matches = re.findall(r'<lora:(.*?):', parameter)
    loras = '_'.join(set(matches))

    #loras = '_'.join(matches)

    if len(loras) >0:
        #print("Lora !")
        return loras
    else:
        return None

def substring(stringtosearch,start_substring,end_substring):

    #start_substring = 'lora hashes: "'
    #end_substring = '"'

    start_index = stringtosearch.find(start_substring)
    end_index = stringtosearch.find(end_substring, start_index + len(start_substring))

    if start_index != -1 and end_index != -1:
        result = stringtosearch[start_index + len(start_substring):end_index]
        #print(result)
    else:
        print("Substrings not found.")
        return None
   
    return result

def move_file_to_meta(filename, subfolder_name):
    file_location = os.path.dirname(filename)
    destination_folder = os.path.join(file_location, subfolder_name)

    subfolder_name = sanitise_path_name(subfolder_name)
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

def move_to_subfolder(path, subfolder):
    # Check if the path is a directory or a file

    last_folder = os.path.basename(os.path.dirname(path))

    if subfolder in last_folder:
        print(path + " directory tree already contains " + subfolder)
        return

    if os.path.isdir(path):
        # Create the subfolder if it doesn't exist
        subfolder_path = os.path.join(path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Move all files in the directory to the subfolder
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                base_name, ext = os.path.splitext(file)
                dest_file = os.path.join(subfolder_path, file)
                count = 1
                while os.path.exists(dest_file):
                    new_name = f"{base_name}_{count}{ext}"
                    dest_file = os.path.join(subfolder_path, new_name)
                    count += 1
                shutil.move(os.path.join(path, file), dest_file)

    elif os.path.isfile(path):
        # Create the subfolder if it doesn't exist
        subfolder_path = os.path.join(os.path.dirname(path), subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Move the file to the subfolder
        base_name, ext = os.path.splitext(os.path.basename(path))
        dest_file = os.path.join(subfolder_path, os.path.basename(path))
        count = 1
        while os.path.exists(dest_file):
            new_name = f"{base_name}_{count}{ext}"
            dest_file = os.path.join(subfolder_path, new_name)
            count += 1
        shutil.move(path, dest_file)

def move_to_fixed_folder_with_group_number(path, filepath,groupid):
    # Check if the path is a directory or a file

    global log_file
    dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base_name, ext = os.path.splitext(filename)

    destination = os.path.join(path,groupid,groupid + '_' + filename)
    path = os.path.join(path,groupid)
    
    if not os.path.isdir(path):
        os.makedirs(path)

    if os.path.isfile(destination):
        # Create the subfolder if it doesn't exist
        count = 1
        while os.path.exists(destination):
            new_name = f"{groupid}_{base_name}_{count}{ext}"
            destination = os.path.join(path, new_name)
            count += 1
        write_to_log(log_file,f"moving {filepath} to {destination}")
        shutil.move(filepath, destination)
    else:
        write_to_log(log_file,f"moving {filepath} to {destination}")
        shutil.move(filepath, destination)


def main(foldertosearch,destination_folder,path_to_style_file):

    global readstyles
    global showcounts
    global writecsv
    global movefiles
    global moveiffilesover
    global log_file
    global comparebymd5
    global comparebytext
    global comparebytextpercentage
    global renamefiles
    global dump_prompt
    global csv_filename
    global csv_filename_neg
    global csv_filename_pos
    global csv_filename_prompts

    # Create a dictionary to store file hashes and corresponding folders
    file_hash_to_folder = {}
    hash_to_files = {}
    hash_list = defaultdict(list)
    new_array = defaultdict(list)
    pos_valuescount = []
    neg_valuescount = []
    neg_values = []

    if readstyles == True:
        if os.path.exists(path_to_style_file):
            stylevars = read_style_to_list(path_to_style_file)
        else:
            readstyles = False

    foldercnt = find_highest_numbered_folder(destination_folder)

    counter = 0
    # Iterate through all files in the foldertosearch and its subdirectories
    for root, dirs, files in os.walk(foldertosearch):
        for filename in files:

            file_path = os.path.join(root, filename)

            if filename.endswith(".png"):
                counter += 1
                print("Processing: " + str(counter))
                parameter = None
                badfile = False
                hasparameters = False
                platform = None
                positiveprompt = None
                negativeprompt = None
                seed = None
                loras = None
                with Image.open(file_path) as img:
                    if hasparameters == False:
                        try:
                            parameter = img.info.get("parameters")
                        except Exception as e:
                            print(f"Error {e}")
                        if parameter is not None:
                            print(filename + " has A1111 metadata.")
                            platform = "A1111"
                            hasparameters = True

                            print(parameter)

                            settings = parse_generation_parameters(parameter)
                            #settings = [item.strip() for item in settings]

                            positiveprompt = ""
                            negativeprompt = ""
                     
                            #result = mytest(parameter)
                            positivepromptarray = settings['Prompt'].split(', ')

                            if 'Template: ' in parameter:
                                print("template\n" + parameter)
                                try:
                                    templatevar = settings['Template'].split(', ')
                                    if str(templatevar) != str(positivepromptarray):
                                        print(f"template and prompt don't match\n{templatevar}\n{positivepromptarray}")
                                except:
                                    print("oops")

                            negativepromptarray = settings['Negative prompt'].split(', ')

                            if negativepromptarray != None:
                                negativepromptarray = sorted(negativepromptarray)

                            if readstyles == True:
                                pos1, neg1 = checkposandnegarrays(stylevars,positivepromptarray,negativepromptarray)
                            else:
                                print("no Styles file")

                            if pos1 != None and neg1 != None:
                                print("Embedded Styles were used")
                                pos1 = sorted(pos1)
                                neg1 = sorted(neg1)
                                pos_valuescount.extend(pos1)
                                neg_valuescount.extend(pos1)

                                positiveprompt = ','.join(pos1)
                                negativeprompt = ','.join(neg1)

                            else:
                                print("No embedded Styles used")
                                if positivepromptarray != None:
                                    positivepromptarray = sorted(positivepromptarray)
                                    pos_valuescount.extend(positivepromptarray)
                                    positiveprompt = ','.join(positivepromptarray)

                                if negativepromptarray != None:
                                    negativepromptarray = sorted(negativepromptarray)                    
                                    negativeprompt = ','.join(negativepromptarray)

                                try:
                                    steps = settings['Steps']
                                except:
                                    steps = None
                                
                                try:
                                    seed = settings['Seed']
                                except:
                                    seed = None

                                try:
                                    model = settings['Model']
                                except:
                                    model = None

                            if 'lora' in parameter.lower():
                                foundlora = False
                                loras1 = None
                                loras2 = None
                                loras3 = None
                                if '<lora:' in positiveprompt.lower():
                                    print('Loras declared in prompt')
                                    loras1 = extract_between_angle_brackets(positiveprompt)
                                    loras1 = '_'.join(loras1)
                                    foundlora = True
                                    print(f"loras1 in prompt: {loras1}")
                                else:
                                    print("No Lora in prompt")
                                # AddNet Module 1: LoRA, AddNet Model 1:
                                # <lora:epiNoiseoffset_v2:0.6>
                                if 'lora' in parameter.lower():
                                    loras2 = getloras(parameter)
                                    if loras2 != None:
                                        foundlora = True
                                        print(f"loras2: {loras2}")
                                if ': LoRA' in parameter:
                                        if'AddNet Model 1' in settings:
                                            loras3 = settings['AddNet Model 1']
                                            if loras3 == '':
                                                print("There should be a Lora but it's missing")
                                            else:
                                                foundlora = True
                                                print(f"loras3: {loras3}")
                                        if'AddNet Model 2' in settings:
                                            loras3 = f"{loras3}_{settings['AddNet Model 2']}"
                                            if loras3 == '':
                                                print("There should be a Lora but it's missing")
                                            else:
                                                foundlora = True
                                                print(f"loras3: {loras3}")
                                
                                if foundlora == False:
                                    print("Investigate possibly")
                                else:
                                    
                                    loras = loras1
                                    if loras == None:
                                        loras = loras2
                                    if loras == None:
                                        loras = loras3
                                    print(f"{loras}")


                        else:
                            print("No parameters")
                    if hasparameters == False:
                        try:
                            parameter = img.info.get("prompt")
                        except Exception as e:
                            print(f"Error {e}")

                        if parameter is not None:
                            #print(filename + " has metadata.")
                            platform = "comfyUI"
                            hasparameters = True
                            badfile = False
                            #settings = parse_generation_parameters(parameter)
                            continue
                            try:
                                json_data = json.loads(parameter)
                                model = json_data.get('1', {}).get('inputs',{}).get('ckpt_name', None).replace('.safetensors','')
                                positiveprompt = json_data.get('1', {}).get('inputs',{}).get('positive', None)
                                negativeprompt = json_data.get('1', {}).get('inputs',{}).get('negative', None)
                                seed = json_data.get('1', {}).get('inputs',{}).get('seed', None)

                                steps = json_data.get('2', {}).get('inputs',{}).get('steps', None)
                                cfg = json_data.get('2', {}).get('inputs',{}).get('cfg', None)
                                sampler = json_data.get('2', {}).get('inputs',{}).get('sampler_name', None)
                            except Exception as e:
                                print(f"Error {file_path}.  {e}.  \nPress a key to continue")
                                input()
                            loras = None

                            print("we don't handle comfyui yet")
                        else:
                            print("No prompt")
                    if hasparameters == False:
                        try:
                            parameter = img.info.get("Software")
                        except Exception as e:
                            print(f"Error {e}")

                        if parameter is not None:
                            if 'NovelAI' in parameter:
                                print("NovelAI Picture")
                                platform = "NovelAI"

                                hasparameters = True
                                badfile = False

                                #settings = parse_generation_parameters(parameter)


                                json_data = json.loads(img.info.get("Comment"))
                                json_string = json.dumps(json_data, separators=(',', ':')).lower()
                                #uc is undesired content i.e. negative prompt
                                steps = json_data['steps']
                                seed = json_data['seed']
                                sampler = json_data['sampler']
                                strength = json_data['strength']
                                noise = json_data['noise']
                                scale = json_data['scale']
                                negativeprompt = json_data['uc']
                                model = None
                                loras = None
                                #model
                                #loras
                                positiveprompt = img.info.get("Description").lower()
                                #print("we don't handle comfyui yet")
                        else:
                            print("No parameters")
                    if hasparameters == False:
                        badfile = True
                if positiveprompt == None: positiveprompt = ""
                if negativeprompt == None: negativeprompt = ""
                if dump_prompt == True and positiveprompt != "":
                    if not os.path.exists(csv_filename_prompts):
                        with open(csv_filename_prompts, 'w', newline='', encoding='utf-8') as csv_file:
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerow(['filename', 'positive prompt', 'negative prompt'])
                            csv_writer.writerows([filename,positiveprompt,negativeprompt])
                    else:
                        with open(csv_filename_prompts, 'a', newline='', encoding='utf-8') as csv_file:
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerows([filename,positiveprompt,negativeprompt])

                if renamefiles == True and hasparameters == True:
                    new_filename = ""
                    if platform is not None:
                        new_filename = f"{new_filename}{platform}_"
                        print ("model is " + platform)
                    else:
                        new_filename = f"{new_filename}noplatform_"

                    if model is not None:
                        new_filename = f"{new_filename}{model}_"
                        print ("model is " + model)
                    else:
                        new_filename = f"{new_filename}nomodel_"

                    if seed is not None:
                        print("Seed is " + str(seed))
                        new_filename = f"{new_filename}{seed}_"
                    else:
                        new_filename = f"{new_filename}noseed_"

                    #new_filename = new_filename + '_' + get_sanitised_download_time(file_path) + '_'
                    # os.path.splitext(filename)[1]
                    if loras != None:
                        loras_sanitised = sanitise_path_name(loras)                    
                        if loras_sanitised != "":
                            new_filename = f"{new_filename}loras_{loras_sanitised}"
                        #else:
                        #    print("uses no loras")

                    new_filename = new_filename + os.path.splitext(filename)[1]
                    new_filename = sanitise_path_name(new_filename)
                    new_item_path = os.path.join(root, new_filename)

                    print(new_item_path)

                    if file_path not in new_item_path:
                        try:
                            shutil.move(file_path, new_item_path)
                        except Exception as e:
                            print(str(e))
                    else:
                        print("doesn't need renaming.  Src and dest are the same: " + file_path + ' ' + new_item_path)
                else:
                    new_item_path = file_path

                if positiveprompt != "":
                        #write_to_log(log_file, new_item_path + " . " + positiveprompt)

                        # Calculate an MD5 hash of the section content
                        if comparebymd5 == True:
                            section_hash = hashlib.md5(positiveprompt.encode()).hexdigest()

                            msg = new_item_path + " . " + section_hash
                            print(msg)
                            write_to_log(log_file, msg)

                            if section_hash in file_hash_to_folder:
                                folder_name = file_hash_to_folder[section_hash]
                            else:
                                folder_name = section_hash
                                file_hash_to_folder[section_hash] = folder_name

                            #hash_list.append([section_hash, filename, new_item_path])
                            hash_list[section_hash].append([new_filename, new_item_path])

                            if section_hash in hash_to_files:
                                hash_to_files[section_hash].append(new_item_path)
                            else:
                                hash_to_files[section_hash] = [new_item_path]

                        elif comparebytext == True:
                            new_array[new_item_path].append(positiveprompt)

            elif filename.endswith(".jpeg") or filename.endswith(".jpg"):
                badfile = True

                with Image.open(file_path) as img:
                    print(str(img.info))
                print(f"don't do anything with {filename}")

            else:
                print(f"Ignoring unsupported filetype: {filename}")
                continue

            if badfile==True:
                    if move_nometa == True:
                        print(filename + " has no metadata.  Moving to nometa subdirectory")
                        move_to_subfolder(file_path,"nometa")
                    else:
                        print("moving files with no metadata is disabled")

    if showcounts == True:
        value_counts = Counter(pos_valuescount)
        # Sort values by occurrence count in decreasing order
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        # Print the values and their occurrence counts
        for value, count in sorted_values:
            print(f'{value}: {count}')
        with open(csv_filename_pos, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['description', 'count'])
            csv_writer.writerows(sorted_values)

        value_counts = Counter(neg_valuescount)
        # Sort values by occurrence count in decreasing order
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        # Print the values and their occurrence counts
        for value, count in sorted_values:
            print(f'{value}: {count}')
        with open(csv_filename_neg, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['description', 'count'])
            csv_writer.writerows(sorted_values)

    # Sort the list by content hash
    #hash_list.sort(key=lambda x: x[0])
    # Create a CSV file to store the results
    if writecsv == True and comparebymd5 ==True :
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Content Hash', 'Count', 'Filename', 'Full Path'])
            sorted_hash_dict = dict(sorted(hash_list.items(), key=lambda item: item[0]))

            for content_hash, data in sorted_hash_dict.items():
                count = len(data)
                for item in data:
                    filename = item[0]
                    full_path = item[1]
                    csv_writer.writerow([content_hash, count, filename, full_path])

    if movefiles == True:
        if comparebytext == True:
            #result = create_word_groups(new_array,comparebytextpercentage,moveiffilesover)
            #looper = 0
#            for i, group in enumerate(result, start=1):
                #looper +=1
#                print(f"Group {i}: {group}")
#                for each in group:
#                    print("group " + str(i) + " of " + str(len(result)))
#                    if len(group) > moveiffilesover:
                    #shouldn't need to do this.  why do I ?
#                        move_to_fixed_folder_with_group_number(destination_folder,each,str(i))
            word_groups = create_word_groups_precentage(new_array,comparebytextpercentage,moveiffilesover)
            #word_groups = create_word_groups_parallel(new_array)
            print("Word Groups:")
            for i, group in enumerate(word_groups, start=1):
                for each in group:
                    print(f"Group {i}: {group}")
                    if foldercnt is not None:
                        final = foldercnt + i
                    else:
                        final = i
                    move_to_fixed_folder_with_group_number(destination_folder,each,str(final))
        elif comparebymd5 ==True:
            for hash, files in hash_to_files.items():
                        if len(files) > moveiffilesover:
                            for file_path in files:
                                folder_name = os.path.join(os.path.dirname(file_path), hash)
                                new_file_path = os.path.join( folder_name, os.path.basename(file_path))
                                if hash in str(os.path.dirname(file_path)):
                                    print("Filename " +str(files) + " already in a directory containing the hash " + hash)
                                    continue
                                else:
                                    if os.path.normpath(file_path) == os.path.normpath(new_file_path):
                                        print("Already in correct location")
                                    else:
                                        os.makedirs(folder_name, exist_ok=True)
                                        try:
                                            os.rename(file_path, new_file_path)
                                        except Exception as e:
                                            print("oops " + str(e))
                                        print(f"Moved: {file_path} to {new_file_path}")
    else:
        print("Not moving files")
    print("Files have been organized into folders.")

root_directory = '/file/to/sort/'
stylefilepath = '/path/to/styles.csv'
sorted_folder = '/file/to/Sorted/'

readstyles = True
showcounts = True
writecsv = True
debug = True
movefiles = False
renamefiles = False
moveiffilesover = 1
comparebymd5 = False
comparebytext=True
dump_prompt = True
comparebytextpercentage=80
useapikey = False
move_nometa = False

if useapikey == True:
    #unused here
    apifile = os.path.join(get_script_path(),"apikey.py")
    if os.path.exists(apifile):
        import apifile

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

log_file = os.path.join(get_script_path(),get_script_name() + '.log')
csv_filename = os.path.join(root_directory, 'hash_results.csv')
csv_filename_neg = os.path.join(root_directory,'neg_parameter_counts.csv')
csv_filename_pos = os.path.join(root_directory,'pos_parameter_counts.csv')
csv_filename_prompts = os.path.join(root_directory,'prompts.csv')
                                
main(root_directory,sorted_folder,stylefilepath)