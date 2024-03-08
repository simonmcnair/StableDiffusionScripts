from PIL import Image
import os

def check_and_rename_image(file_path):
    try:
        with Image.open(file_path) as img:
            # Get the actual format of the image
            actual_format = img.format.lower()

        # Get the original extension
        _, original_extension = os.path.splitext(file_path)

        # Map the actual format to the correct extension
        format_to_extension = {'jpeg': '.jpg', 'jpg': '.jpg', 'png': '.png'}

        if actual_format in format_to_extension:
            new_extension = format_to_extension[actual_format]

            # Rename the file only if the extension needs to be changed
            if new_extension != original_extension.lower():
                new_file_path = file_path.replace(original_extension, new_extension)
                os.rename(file_path, new_file_path)
                print(f'Renamed: {file_path} -> {new_file_path}')
    except Exception as e:
        # Handle cases where the file is not a valid image
        print(f'Error processing {file_path}: {e}')

def recursively_check_and_rename(root_path):
    for foldername, subfolders, filenames in os.walk(root_path):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            
            # Check if the file has a .png or .jpg extension
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                check_and_rename_image(file_path)

# Example usage:
root_directory = 'X:/dif/stable-diffusion-webui-docker/data/models/Lora'
recursively_check_and_rename(root_directory)
