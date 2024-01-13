import os
import csv
from collections import defaultdict

def find_duplicate_words_in_directory(root_folder):
    # Dictionary to store words and the files in which they appear
    word_occurrences = defaultdict(list)

    for folder_name, _, filenames in os.walk(root_folder):
        for filename in filenames:
            print("processing ", filename)
            file_path = os.path.join(folder_name, filename)

            if "txt" in filename:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line_number, line in enumerate(file, start=1):
                        print(line.strip().lower())
                        # Append the file to the list of files where the word occurs
                        word_occurrences[line.strip().lower()].append(file_path)

    # Filter out words that occur only once
    word_occurrences = {word: files for word, files in word_occurrences.items() if len(files) > 1}
   
    return word_occurrences

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

    print(f"CSV file '{output_file}' generated.")
