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

def read_style_to_list(file_path):
    data_array = []

    #with open(file_path, 'r', encoding='utf-8') as f:
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            name = row['name'].lower()
            pos_prompt = row['prompt'].lower()
            neg_prompt = row['negative_prompt'].lower()

            if not pos_prompt == '' or not neg_prompt == '':
                data_array.append(row)
            else:
                print("ignoring delimiter " + name)


    return data_array


def checkposandneg(data_dict, string1, string2):
    # Assuming the fields are named [string1example], [string2example], and [name]
    field1_name = f"prompt"
    field2_name = f"negative_prompt"
    name_field = "name"

    string1 = string1.lower()
    string2 = string2.lower()
    posprompt = []
    negprompt = []
    bothprompt = []
    both = None
    field1 = None
    field2 = None
    poscnt = 0
    negcnt = 0
    bothcnt = 0
    #print(type(data_dict))
    for entry in data_dict:
       # print(entry)
        posline = entry[field1_name].lower()
        negline = entry[field2_name].lower()
        if string1 in posline and string2 in negline and posline != '' and negline != '':
            both = entry.get(name_field)
            bothprompt.append(entry.get(name_field))
            print("both.  ",posline,negline,both)
            bothcnt +=1
        elif string2 in negline and negline != '':
            field2 =  entry.get(name_field)
            #test = negline
            negprompt.append(entry.get(name_field))
            print("field2",negline,field2)
            negcnt +=1
        elif string1 in posline and posline != '':
            field1 = entry.get(name_field)
            #test = posline
            posprompt.append(entry.get(name_field))
            print("field1",posline,field1)
            poscnt +=1

    print("Counters are both,pos,neg" ,bothcnt,poscnt,negcnt )    
    return bothprompt,posprompt,negprompt,bothcnt,poscnt,negcnt

def extract_text_after2(list_obj, text):
    for element in list_obj:
        if element.strip().lower().startswith(text.lower() + ":"):
            return element.split(":")[-1].strip()
    return None

def getmodel(mystr):
    test = parameter.split(",")
    try:
        res = extract_text_after2(test,"Model")
    except:
        return None
    return res

def getseed(mystr):
    test = parameter.split(",")
    try:
        res = extract_text_after2(test,"Seed")
    except:
        return None
    return res
    
def findtags(inputstring):
    inputstring = inputstring.replace("\r", "")
    inputstring = inputstring.replace("\n", "")
    inputstring = [substring.strip() for substring in inputstring.split(',')]
    print(inputstring)
    return

def get_sanitized_download_time(filepath):
    # Get the modification time of the file
    mtime = os.path.getmtime(filepath)
    # Convert the modification time to a datetime object
    dt = datetime.fromtimestamp(mtime)
    # Format the datetime as a string in the format YYYY-MM-DD_HH-MM-SS
    dt_string = dt.strftime("%Y-%m-%d_%H-%M-%S")
    # Replace any spaces or colons with underscores
    sanitized_dt_string = dt_string.replace(" ", "_").replace(":", "_")
    # Return the sanitized datetime string
    return sanitized_dt_string

def getnegprompt(parameter):
    if "Negative prompt" in parameter:
        negprompt = parameter.split("Negative prompt", 1)[1]
    else:
        negprompt = re.split(r'[\r\n]+', parameter)[1]
        #negprompt = re.split(r'[\r\n]Steps', parameter)
    if negprompt.startswith(": "):
        # Remove the first occurrence of ":"
        negprompt = negprompt[2:] 
    return negprompt
def getposprompt(parameter):

    if "Negative prompt" in parameter:
        parts = parameter.split("Negative prompt", 1)
    else:
        parts = re.split(r'[\r\n]+', parameter)
        
    if len(parts) >= 1:
        # Use regular expressions to extract the 'parameter' section
            section_content = parts[0]
            section_content = section_content.replace('\r\n', '')
            section_content = section_content.replace('\r', '').replace('\n', '')
    return section_content

def getloras(parameter):

    matches = re.findall(r'lora:(.*?):', parameter)
    Loras = '_'.join(set(matches))

    #Loras = '_'.join(matches)

    if len(Loras) >0:
        #print("Lora !")
        return Loras
    else:
        return ""

def write_to_log(log_file, message):
    try:
        with open(log_file, 'a') as file:
            file.write(message + '\n')
    except Exception as e:
        print(f"Error writing to the log file: {e}")

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

# Root directory to start the recursive search
root_directory = 'Z:/Pron/Pics/stable-diffusion/consolidated/AmyWong'
#root_directory = 'X:/dif/stable-diffusion-webui-docker/output/txt2img/Newfolder'

log_file = os.path.join(root_directory,"my_log.txt")

# Create a dictionary to store file hashes and corresponding folders
file_hash_to_folder = {}
hash_to_files = {}
hash_list = defaultdict(list)
pos_values = []
neg_values = []


# Iterate through all files in the root_directory and its subdirectories
for root, dirs, files in os.walk(root_directory):
    for filename in files:
        hasparameters = False

        file_path = os.path.join(root, filename)

        if filename.endswith(".png"):
            with Image.open(file_path) as img:
                try:
                    parameter = img.info.get("parameters")
                    if parameter is not None:
                        #print(filename + " has metadata.")
                        hasparameters = True
                        badfile = False
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

            model = ""
            seed = ""
            loras = ""
            new_filename = ""
            positiveprompt = ""
            negativeprompt = ""

            model = getmodel(parameter)
            seed = getseed(parameter)
            positiveprompt = getposprompt(parameter)
            negativeprompt = getnegprompt(parameter)
            loras = getloras(parameter)

            if positiveprompt != "":
                posvalues = positiveprompt.split(",")
                posvalues = [substring.strip() for substring in posvalues]
                #list of positive prompt entries
                #pos_values.extend(posvalues)

            if negativeprompt != "":
                negvalues = negativeprompt.split(",")
                negvalues = [substring.strip() for substring in negvalues]
                #list of positive prompt entries
                #negvalues.extend(neg_values)
           
            if model is not None:
                new_filename = model + '_'
            else:
                new_filename = "nomodel_"

            if seed is not None:
                new_filename = new_filename + seed  + '_'
            else:
                new_filename = new_filename + "noseed_"


            new_filename = new_filename + '_' + get_sanitized_download_time(file_path) + '_'
            # os.path.splitext(filename)[1]

            if loras != "":
                new_filename = new_filename + 'Loras_' + loras + '_'
            #else:
            #    print("uses no Loras")

            new_filename = new_filename + os.path.splitext(filename)[1]
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
                # Use regular expressions to extract the 'parameter' section

                    write_to_log(log_file, new_item_path + " . " + positiveprompt)

                    # Calculate an MD5 hash of the section content
                    section_hash = hashlib.md5(positiveprompt.encode()).hexdigest()

                    msg = new_item_path + " . " + section_hash
                    print(msg)
                    write_to_log(log_file, msg)

                    if section_hash in file_hash_to_folder:
                        folder_name = file_hash_to_folder[section_hash]
                    else:
                        #new_uuid = uuid.uuid4()
                        # Convert the UUID to a string
                        #folder_name = str(new_uuid)
                        folder_name = section_hash
                        file_hash_to_folder[section_hash] = folder_name

                    #hash_list.append([section_hash, filename, new_item_path])
                    hash_list[section_hash].append([new_filename, new_item_path])


                    if section_hash in hash_to_files:
                        hash_to_files[section_hash].append(new_item_path)
                    else:
                        hash_to_files[section_hash] = [new_item_path]

        if badfile==True:
                print(filename + " has no metadata.  Moving to nometa subdirectory")
                move_to_subfolder(new_item_path,"nometa")

showcounts = True

if showcounts == True:
    value_counts = Counter(pos_values)

    # Sort values by occurrence count in decreasing order
    sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)

    # Print the values and their occurrence counts
    for value, count in sorted_values:
        print(f'{value}: {count}')
    csv_filename = os.path.join(root_directory,'parameter_counts.csv')
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['description', 'count'])
        csv_writer.writerows(sorted_values)

# Sort the list by content hash
#hash_list.sort(key=lambda x: x[0])
writecsv = False
# Create a CSV file to store the results
if writecsv == True:
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

movefiles = True
moveiffilesover = 1
#renamefiles = False
for hash, files in hash_to_files.items():
 
            if movefiles == True:
                if len(files) > moveiffilesover:

                    for file_path in files:
                       
                       
                        folder_name = os.path.join(os.path.dirname(file_path), hash)
                        new_file_path = os.path.join( folder_name, os.path.basename(file_path))
                            
                        
                        if hash in str(os.path.dirname(file_path)):
                        #if hash in str(os.path.basename(file_path)) and hash in str(os.path.dirname(file_path)):
                            print("Filename " +str(files) + " already in a directory containing the hash " + hash)
                            continue
                        else:
                            if os.path.normpath(file_path) == os.path.normpath(new_file_path):
                                print("Already in correct location")
                            else:
                                os.makedirs(folder_name, exist_ok=True)
                                os.rename(file_path, new_file_path)
                                print(f"Moved: {file_path} to {new_file_path}")
            #elif renamefiles == True:
            #    for file_path in files:
            #        if hash in str(file_path):
            #            print("Filename " +str(files) + "already contains hash " + hash)
            #            continue                    
            #        new_file_path = os.path.join( os.path.dirname(file_path), hash + '_' + os.path.basename(file_path))
            #        if file_path != new_file_path:
            #            try:
            #                os.rename(file_path, new_file_path)
            #            except Exception as e:
            #                print(str(e))


print("Files have been organized into folders.")
