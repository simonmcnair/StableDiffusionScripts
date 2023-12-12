import os
from collections import Counter
from PIL import Image
import csv

# Define the output file
output_file = "metadata.csv"

# Initialize the counters for positive, negative, and extra prompts
positive_counter = Counter()
negative_counter = Counter()
extra_counter = Counter()


def process_string(input_string,var):
    # Replace ", " with ","
    input_string = input_string.replace(", ", ",")
    # Replace " ," with ","
    input_string = input_string.replace(" ,", ",")
    # Remove excess spaces from beginning and end of string
    input_string = input_string.strip()
    # Replace "\n\n" with "\n"
    input_string = input_string.replace("\n\n", "\n")
    # Split string into an array using "\n" as delimiter
    output_array = input_string.split(var)
    return output_array

def remove_negative_prompt(input_string,removal):
    output_string = input_string.replace(removal, '').strip()
    return output_string

def remove_empty_dict_values(input_dict):
    output_dict = {k: v for k, v in input_dict.items() if v}
    return output_dict

def remove_empty_values(input_list):
    output_list = [value for value in input_list if value != ""]
    return output_list

#def find_prompt(input_list, prompt):
#    try:
#        index = input_list.index(prompt)
#        return index
#    except ValueError:
#        return None
import csv

def write_array_to_csv(array, filename, header=None):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if header:
            writer.writerow(header)
        for row in array:
            writer.writerow(row)


def sort_dictionary(dictionary):
    sorted_keys = sorted(dictionary.keys(), key=lambda x: (x.split(":")[0], -dictionary[x]))
    sorted_dict = {key: dictionary[key] for key in sorted_keys}
    return sorted_dict

def prefix_list(input_list, prefix):
    output_list = [prefix + str(value) for value in input_list]
    return output_list

def find_prompt(input_list, search_text):
    try:
        for i, value in enumerate(input_list):
            if search_text in value:
                return i
        return -1
    except ValueError:
        return None

def aggregate_data(destination, addition):
    for value in addition:
        if value in destination:
            destination[value] += 1
        else:
            destination[value] = 1
    return destination

destinationarr = {}
# Loop over all PNG files in the current working directory
for filename in os.listdir():
    if filename.endswith(".png"):
        # Open the image and extract the metadata
        print("Processing " + filename)
        with Image.open(filename) as img:
            metadata = img.info.get("parameters", "")

        metadata = process_string(metadata,"\n")

        positiveprompt = process_string(metadata[0],",")
        positiveprompt = remove_empty_values(positiveprompt)
        positiveprompt = prefix_list(positiveprompt,"Positive Prompt:")

        res = find_prompt(metadata,'Negative prompt: ')
        if res is None:
            print("The string 'Negative prompt' was not found in the list.")
        else:
            print("The index of the first occurrence of 'Negative prompt' is:", res)
                        
            if 'Negative prompt:' in metadata[res]:
                negativeprompt = remove_negative_prompt(metadata[res],'Negative prompt:')
                negativeprompt = process_string(negativeprompt,",")
                negativeprompt = remove_empty_values(negativeprompt)
            else:
                print("No negative prompt")
        negativeprompt = prefix_list(negativeprompt,"Negative Prompt:")


        res = find_prompt(metadata,'Steps: ')
        if res is None:
            print("The string 'Steps' was not found in the list.")
        else:
            print("The index of the first occurrence of 'Steps' is:", res)
            extra = metadata[res].split(",")

        result = positiveprompt + negativeprompt + extra
   
            #if 'Negative prompt:' in metadata[res]:
            #    negativeprompt = remove_negative_prompt(metadata[res],'Negative prompt:')
            #    negativeprompt = process_string(negativeprompt,",")
            #    negativeprompt = remove_empty_values(negativeprompt)
            #else:
            #    print("No negative prompt")
    sorte_dictionary = aggregate_data(destinationarr,result)    


sorte_dictionary = sort_dictionary(sorte_dictionary)
    # Sort the data by count in descending order

#destinationarr =     sorted_dict = dict(sorted(destinationarr.items(), key=lambda x: (x[0].split(":")[0], -int(x[0].split(",")[1]))))
 
head = {"setting", "description","count"}

write_array_to_csv(sorte_dictionary,output_file, head)
# Write the data to the output file



