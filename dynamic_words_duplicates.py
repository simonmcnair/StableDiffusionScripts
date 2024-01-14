import os
import csv
from collections import defaultdict
import pandas as pd

def tell_me_about(s): return (type(s), s)

def find_non_utf8_characters(input_string):
    non_utf8_characters = []
    #print("string was " + input_string)
    return input_string.encode('latin-1').decode('utf-8')


#    for char in input_string:
#        print(char)
#        try:
#            #char.encode('utf-8')
#            char = repr(char).encode('utf-8')
#        except UnicodeEncodeError:
#            non_utf8_characters.append(char)
#        except Exception as e:
#            print("exception" + str(e))
#            print("exception" + str(e))
#    print("string is now " + input_string)
    

#    return non_utf8_characters

def find_invalid_utf8_bytes(input_bytes):
    invalid_bytes = []

    try:
        input_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        invalid_bytes.append((e.start, e.end))
    except Exception as e:
        print("oops")
        print("oops")
    
    return invalid_bytes

def find_duplicate_words_in_directory(root_folder):
    # Dictionary to store words and the files in which they appear
    word_occurrences = defaultdict(list)
    test = defaultdict(list)

    for folder_name, _, filenames in os.walk(root_folder):
        for filename in filenames:
            print("processing ", filename)
            file_path = os.path.join(folder_name, filename)
            if "txt" in filename:
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        for line_number, line in enumerate(file, start=1):
                            print(line.strip().lower())
                            # Append the file to the list of files where the word occurs
                            word_occurrences[line.strip().lower()].append(file_path)
                            test[file_path].append(line.strip().lower())
                    except Exception as e:
                        print(tell_me_about(line))
                        try:
                            ret = find_non_utf8_characters(line)
                            word_occurrences[ret.strip().lower()].append(file_path)
                            print("non utf-8 character is " + str(ret))
                        except:
                            print("Failed")
    # Filter out words that occur only once
    word_occurrences = {word: files for word, files in word_occurrences.items() if len(files) > 1}
    word_occurrences2 = {word: files for word, files in test.items() if len(files) > 1}
   
    return word_occurrences

def remove_duplicate_lines_interactively(file_word_mapping):
    for file_path, words in file_word_mapping.items():
        print(f"Processing file: {file_path}")

        unique_words = list(set(words))
        lines_to_write = []

        for word in unique_words:
            if words.count(word) > 1:
                print(f"\nDuplicate word found: {word}")
                print(f"Original lines: {words}")

                # Prompt the user to choose which file retains the word
                choice = input(f"Choose the file to retain the word ('{file_path}' or another file): ")

                if choice != file_path:
                    # Remove the word from other files
                    for other_file_path in file_word_mapping.keys():
                        if choice == other_file_path:
                            file_word_mapping[other_file_path] = [w for w in file_word_mapping[other_file_path] if w != word]

                # Build lines to write for the current file
                lines_to_write.extend(line for line in open(file_path, 'r', encoding='utf-8', errors='ignore') if word not in line)

        # Update the actual file with the modified content
        with open(file_path, 'w', encoding='utf-8', errors='ignore') as file:
            file.writelines(lines_to_write)

def sort_csv(input_file, output_file, sort_column):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        sorted_rows = sorted(reader, key=lambda row: row[sort_column])

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        fieldnames = reader.fieldnames
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)

def create_file_word_mapping(word_occurrences):
    file_word_mapping = defaultdict(list)

    for word, files in word_occurrences.items():
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as file:
                words = [word.strip() for line in file for word in line.split()]
                file_word_mapping[file_path].extend(words)

    return file_word_mapping

def write_csv(output_file, word_occurrences):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Word', 'Files'])

        for word, files in word_occurrences.items():
            # Remove duplicates from the list of files
            unique_files = list(set(files))
            csv_writer.writerow([word] + unique_files)

if __name__ == "__main__":
    root_folder = "X:/dif/stable-diffusion-webui-docker/data/config/auto/extensions/sd-dynamic-prompts/wildcards"
    output_file = "X:/dif/stable-diffusion-webui-docker/data/config/auto/extensions/sd-dynamic-prompts/wildcards/duplicate_words.csv"

    word_occurrences = find_duplicate_words_in_directory(root_folder)
    write_csv(output_file, word_occurrences)
    sort_csv(output_file,"X:/dif/stable-diffusion-webui-docker/data/config/auto/extensions/sd-dynamic-prompts/wildcards/duplicate_words2.csv",'Word')

    print(f"CSV file '{output_file}' generated.")
