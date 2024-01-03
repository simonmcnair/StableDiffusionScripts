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

def create_word_groups(filepath_strings,percentagesimilar=90,groupifover=1):
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

def properwaytogetPromptfield(mystr,fieldtoretrieve=''):

    positivepromptfound = False
    negativepromptfound = False
    extrasfound = False

    if fieldtoretrieve not in mystr:
        print(fieldtoretrieve + " not found in " + mystr)  
        return None
    
    if 'steps:' in mystr:
        extrasfound = True

    mystr = remove_spaces_around_comma(mystr)
    mystr = mystr.replace("  ", " ")

    if 'negative prompt:' in mystr and fieldtoretrieve=='':
        test = mystr.split("\n")
        negativepromptfound = True
    elif 'negative prompt:' in mystr:
        test = mystr.split("\n")
        negativepromptfound = True
    else:
        test = mystr.split("\n")

    if 'lora' in mystr:
        #print("Lora")
        mystr = ensure_comma_before_lt(mystr)
    
    ret = None
    #if negativepromptfound == False and positivepromptfound == False and fieldtoretrieve=='' or fieldtoretrieve == 'negative prompt':
    #    print("neg or positive no neg or pos prompt")
    #        return None
        

    if fieldtoretrieve == '':
        if test[0].startswith("steps:"):
            print("No positive prompt.  It would have been parsed before steps if it had existed.")
            return None
        else:
            ret = [substring.strip() for substring in test[0].split(',')]
            if '<lora' in ret:
                for each in ret:
                    if '<lora' == each:
                            print("check" + str(each))
                    elif '<lora' in each:
                            print("seems fine " + str(each))

    else:
            for each in test:
                if fieldtoretrieve in ['steps','sampler','cfg scale','seed','model']:
                    each = each.replace('\n','')
                    if fieldtoretrieve in each:
                        #test = each.split([',','\n'])
                        mystr = [substring.strip() for substring in each.split(',')]
                        return extract_text_after2(mystr,fieldtoretrieve)
                else:
                    temp = each.split(':',1)
                    if temp[0] == fieldtoretrieve:
                        #subsplit = temp[1].split(",")
                        ret = [substring.strip() for substring in temp[1].split(',')]
                        break


    if ret != None:
        ret = [item for item in ret if item != ""]

    return ret

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
    Loras = '_'.join(set(matches))

    #Loras = '_'.join(matches)

    if len(Loras) >0:
        #print("Lora !")
        return Loras
    else:
        return ""

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


def main():

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
    global sorted_folder

    # Create a dictionary to store file hashes and corresponding folders
    file_hash_to_folder = {}
    hash_to_files = {}
    hash_list = defaultdict(list)
    new_array = defaultdict(list)
    pos_valuescount = []
    neg_valuescount = []
    neg_values = []

    if readstyles == True:
        if os.path.exists(stylefilepath):
            stylevars = read_style_to_list(stylefilepath)
        else:
            readstyles = False

    foldercnt = find_highest_numbered_folder(sorted_folder)

    counter = 0
    # Iterate through all files in the root_directory and its subdirectories
    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            hasparameters = False

            file_path = os.path.join(root, filename)

            if filename.endswith(".png"):
                counter += 1
                print("Processing: " + str(counter))
                with Image.open(file_path) as img:
                    try:
                        parameter = img.info.get("parameters")
                        if parameter is not None:
                            #print(filename + " has metadata.")
                            hasparameters = True
                            badfile = False
                            parameter = parameter.lower()
                        else:
                            #print("PNG with no metadata")
                            try:
                                parameter = img.info.get("prompt")
                                if parameter is not None:
                                    #print(filename + " has metadata.")
                                    hasparameters = True
                                    badfile = False
                                    print("we don't handle comfyui yet")
                                    continue
                                else:
                                    #print("PNG with no metadata")
                                    badfile = True
                            except:
                                badfile = True
                    except:
                        badfile = True
            elif filename.endswith(".jpeg") or filename.endswith(".jpg"):
                badfile = True
            else:
                print("Ignoring unsupported filetype: " + filename)
                continue

            if hasparameters==True:

               # test = prompt_parser.parse_prompt_attention(parameter)
               # test2 = generation_parameters_copypaste.parse_generation_parameters(parameter)

                positiveprompt = ""
                negativeprompt = ""


                loras = extract_between_angle_brackets(parameter)
                
                if len(loras) > 0 :
                    for lora in loras:
                        print("lora name: " + lora)
                        if 'lora hashes: "' in parameter:
                            print("contains multiple Lora Hashes")
                            
                            result =  substring(parameter,'lora hashes: "','"')
                            allloras = result.split(",")
                            for lor in allloras:
                                lorarray = lor.split(": ")
                                lorname = lorarray[0]
                                lorhash = lorarray[1]
                                print("found lora " + lorname + " with hash " + lorhash)

                            #, adetailer version: 23.11.1, lora hashes: "add_detail: 7c6bad76eb54, add_detail: 7c6bad76eb54", ti hashes: "negative_hand-neg: 73b524a2da12", version: v1.7.0'
                        elif 'lora hashes: ' in parameter:
                            print("contains single Lora Hash")
                            
                            result =  substring(parameter,'lora hashes: ',',')
                            allloras = result.split(",")
                            for lor in allloras:
                                lorarray = lor.split(": ")
                                lorname = lorarray[0]
                                lorhash = lorarray[1]
                                print("found lora " + lorname + " with hash " + lorhash)
                        
                        else:
                            print("No Loras found")

                if 'template:' in parameter:
                    print("template\n" + parameter)
                    positivepromptarray = properwaytogetPromptfield(parameter,"template")
                else:
                    positivepromptarray = properwaytogetPromptfield(parameter,"")

                negativepromptarray = properwaytogetPromptfield(parameter,"negative prompt")

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

                if renamefiles == True:
                    model = ""
                    seed = ""
                    new_filename = ""
                    steps = properwaytogetPromptfield(parameter,"steps")

                    model = properwaytogetPromptfield(parameter,"model")
                    if model is not None:
                        new_filename = model + '_'
                        print ("model is " + model)
                    else:
                        new_filename = "nomodel_"

                    seed = properwaytogetPromptfield(parameter,"seed")
                    if seed is not None:
                        print("Seed is " + str(seed))
                        new_filename = new_filename + seed  + '_'
                    else:
                        new_filename = new_filename + "noseed_"

                    #new_filename = new_filename + '_' + get_sanitised_download_time(file_path) + '_'
                    # os.path.splitext(filename)[1]

                    loras = ""
                    loras = sanitise_path_name(getloras(parameter))                    
                    if loras != "":
                        new_filename = new_filename + 'Loras_' + loras + '_'
                    #else:
                    #    print("uses no Loras")

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

            if badfile==True:
                    print(filename + " has no metadata.  Moving to nometa subdirectory")
                    move_to_subfolder(file_path,"nometa")

    if showcounts == True:
        value_counts = Counter(pos_valuescount)
        # Sort values by occurrence count in decreasing order
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        # Print the values and their occurrence counts
        for value, count in sorted_values:
            print(f'{value}: {count}')
        csv_filename = os.path.join(root_directory,'pos_parameter_counts.csv')
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['description', 'count'])
            csv_writer.writerows(sorted_values)

        value_counts = Counter(neg_valuescount)
        # Sort values by occurrence count in decreasing order
        sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        # Print the values and their occurrence counts
        for value, count in sorted_values:
            print(f'{value}: {count}')
        csv_filename = os.path.join(root_directory,'neg_parameter_counts.csv')
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['description', 'count'])
            csv_writer.writerows(sorted_values)

    # Sort the list by content hash
    #hash_list.sort(key=lambda x: x[0])
    # Create a CSV file to store the results
    if writecsv == True and comparebymd5 ==True :
        csv_filename = os.path.join(root_directory, 'hash_results.csv')
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
#                        move_to_fixed_folder_with_group_number(sorted_folder,each,str(i))

            word_groups = create_word_groups(new_array)
            #word_groups = create_word_groups_parallel(new_array)
            print("Word Groups:")
            for i, group in enumerate(word_groups, start=1):
                for each in group:
                    print(f"Group {i}: {group}")
                    if foldercnt is not None:
                        final = foldercnt + i
                    else:
                        final = i
                    move_to_fixed_folder_with_group_number(sorted_folder,each,str(final))
                    

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

    print("Files have been organized into folders.")




root_directory = '/file/to/sort/'
stylefilepath = '/path/to/styles.csv'
sorted_folder = 'file/to/Sorted/'


readstyles = True
showcounts = True
writecsv = True
debug = True
movefiles = True
renamefiles = True
moveiffilesover = 1
comparebymd5 = False
comparebytext=True
comparebytextpercentage=90
useapikey = False

if useapikey == True:
    #unused here
    apifile = os.path.join(root_directory,"apikey.py")
    if os.path.exists(apifile):
        import apifile

localoverridesfile = os.path.join('.', "localoverridesfile_" + get_script_name() + '.py')

if os.path.exists(localoverridesfile):
    exec(open(localoverridesfile).read())
    #api_key = apikey
    #print("API Key:", api_key)
else:
    print("No local overrides.")


log_file = os.path.join(root_directory,get_script_name() + '.log')
main()