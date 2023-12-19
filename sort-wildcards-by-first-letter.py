import os
import os
import concurrent.futures

def process_files_concurrently(root_directory):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        file_paths = []
        for foldername, subfolders, filenames in os.walk(root_directory):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                file_paths.append(file_path)

        executor.map(sort_file_by_first_letter, file_paths, file_paths)

def sort_file_by_first_letter(input_file, output_file):
    print("processing " + input_file)
    if  os.path.splitext(input_file)[1] == '.txt':
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        # Sort lines based on the first letter
        sorted_lines = sorted(lines, key=lambda x: x[0].lower())

        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.writelines(sorted_lines)

# Example usage
#basefolder = 'C:/Users/Simon/Desktop/'
basefolder = 'X:/dif/stable-diffusion-webui-docker/data/config/auto/extensions/sd-dynamic-prompts/wildcards'
#filename = 'photography-angles.txt'  # Replace with your input file name

#path = os.path.join(basefolder,filename)

#sort_file_by_first_letter(path, path)
while True:

    user_input = input("Are you sure you want to alphabetically sort the files in the directory " + basefolder + "  (y/n): " ).lower()

    if user_input == 'y':
        # Add your code for continuing here
        print("Continuing...")
        process_files_concurrently(basefolder)
        break

    elif user_input == 'n':
        print("Exiting...")
        break  # Exit the loop
    else:
        print("Invalid input. Please enter 'y' to continue or 'n' to exit.")


