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


def create_file_word_mapping(word_occurrences):
    file_word_mapping = defaultdict(list)

    for word, files in word_occurrences.items():
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
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

    print(f"CSV file '{output_file}' generated.")
