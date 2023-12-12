import os
import re
import hashlib
from datetime import datetime
from PIL import Image
#import uuid
import csv
from collections import defaultdict
from collections import Counter

def extract_values_from_parameter(parameter):
    # Use regular expressions to extract values from the parameter section
    values = parameter.split(",")
    return values

def write_to_log(log_file, message):
    try:
        with open(log_file, 'a') as file:
            file.write(message + '\n')
    except Exception as e:
        print(f"Error writing to the log file: {e}")


# Root directory to start the recursive search
#root_directory = 'C:/Users/Simon/Downloads/stable-diffusion/consolidated/Sort/'
root_directory = 'X:/dif/stable-diffusion-webui-docker/output/txt2img/'
log_file = root_directory + "my_log.txt"

# Create a dictionary to store file hashes and corresponding folders
file_hash_to_folder = {}
hash_to_files = {}
hash_list = defaultdict(list)
all_values = []


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

                    else:
                        #print("PNG with no metadata")
                        badfile = True
                except:
                    badfile = True
        elif filename.endswith(".jpeg") or filename.endswith(".jpg"):
            badfile = True
        else:
            print("Ignoring unsupported filetype: " + filename)
            continue

        if hasparameters==True:
            if "Negative prompt" in parameter:
                parts = parameter.split("Negative prompt", 1)
            else:
                parts = re.split(r'[\r\n]+', parameter)
                
            if len(parts) >= 1:
                # Use regular expressions to extract the 'parameter' section
                    section_content = parts[0]

                    section_content = section_content.replace('\r\n', '')

                    section_content = section_content.replace('\r', '').replace('\n', '')

                    print(file_path + " . " + section_content)

                    values = extract_values_from_parameter(section_content)
                    values = [substring.strip() for substring in values]

                    all_values.extend(values)

                    write_to_log(log_file, file_path + " . " + section_content)

                    # Calculate an MD5 hash of the section content
                    section_hash = hashlib.md5(section_content.encode()).hexdigest()

                    msg = file_path + " . " + section_hash
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

                    #hash_list.append([section_hash, filename, file_path])
                    hash_list[section_hash].append([filename, file_path])


                    if section_hash in hash_to_files:
                        hash_to_files[section_hash].append(file_path)
                    else:
                        hash_to_files[section_hash] = [file_path]

showcounts = True

if showcounts == True:
    value_counts = Counter(all_values)

    # Sort values by occurrence count in decreasing order
    sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)

    # Print the values and their occurrence counts
    for value, count in sorted_values:
        print(f'{value}: {count}')
    csv_filename = root_directory + 'parameter_counts.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['description', 'count'])
        csv_writer.writerows(sorted_values)

# Sort the list by content hash
#hash_list.sort(key=lambda x: x[0])
writecsv = False
# Create a CSV file to store the results
if writecsv == True:
    csv_filename = root_directory + 'hash_results.csv'
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
moveiffilesover = 3
renamefiles = False
for hash, files in hash_to_files.items():
 
            if movefiles == True:
                if len(files) > moveiffilesover:

                    for file_path in files:
                       
                       
                        folder_name = os.path.join(os.path.dirname(file_path), hash)
                        new_file_path = os.path.join( folder_name, os.path.basename(file_path))
                            
                        
                        if hash in str(os.path.basename(file_path)) and hash in str(os.path.dirname(file_path)):
                            print("Filename " +str(files) + " and directory already contain hash " + hash)
                            continue
                        else:
                            if os.path.normpath(file_path) == os.path.normpath(new_file_path):
                                print("Already in correct location")
                            else:
                                os.makedirs(folder_name, exist_ok=True)
                                os.rename(file_path, new_file_path)
                                print(f"Moved: {file_path} to {new_file_path}")
            elif renamefiles == True:
                for file_path in files:
                    if hash in str(file_path):
                        print("Filename " +str(files) + "already contains hash " + hash)
                        continue                    
                    new_file_path = os.path.join( os.path.dirname(file_path), hash + '_' + os.path.basename(file_path))
                    if file_path != new_file_path:
                        try:
                            os.rename(file_path, new_file_path)
                        except Exception as e:
                            print(str(e))


print("Files have been organized into folders.")
