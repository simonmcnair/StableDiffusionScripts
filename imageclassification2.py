__author__='Simon McNair'

#python3.11 -m venv venv
#source ./venv/bin/activate

# install torch with GPU support for example:
#pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
# install clip-interrogator
#pip install clip-interrogator
#pip install clip-interrogator==0.5.4
# or for very latest WIP with BLIP2 support
#pip install clip-interrogator==0.6.0

#pip install numpy
#pip install huggingface_hub
#pip install onnxruntime
#pip install pandas
#pip install opencv
#pip install opencv-python
#pip install keras
#pip install tensorflow
#sudo apt-get install python3-tk

#importing libraries
import os
#import glob
from PIL.ExifTags import TAGS
from PIL import Image, ImageTk
from clip_interrogator import Config, Interrogator, list_clip_models
import platform
from transformers import BlipProcessor, BlipForConditionalGeneration

import re
#import threading
from pathlib import Path


import tkinter as tk

def get_operating_system():
    system = platform.system()
    return system


def get_script_name():
    # Use os.path.basename to get the base name (script name) from the full path
    #basename = os.path.basename(path)
    return Path(__file__).stem
    #return os.path.basename(__file__)

def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))

def prepend_string_to_filename(fullpath, prefix):
    # Split the full path into directory and filename
    directory, filename = os.path.split(fullpath)

    # Prepend the prefix to the filename
    new_filename = f"{prefix}{filename}"

    # Join the directory and the new filename to get the updated full path
    new_fullpath = os.path.join(directory, new_filename)

    return new_fullpath


def modify_exif_tags(filename, tags, command, new_value=None, tagname= None):
    # Check if the file exists
    if os.path.exists(filename):
        # Open the image

        original_mtime = os.path.getmtime(filename)
        original_atime = os.path.getatime(filename)

        image = Image.open(filename)

        # Get the Exif data
        exifdata = image.getexif()

        # Use a custom tag (you can modify this based on your requirements)
        found = False
        if tagname is not None:
                for pil_tag, pil_tag_name in TAGS.items():
                    if pil_tag_name == tagname:
                        custom_tag = hex(pil_tag_name)
                        print("using " + tagname + " for tag")
                        found = True
                        break
        if found == False or tagname == None:
            # 40094:0x9C9E:'XPKeywords'
            print("using XPKeywords for tag")
            custom_tag = 0x9C9E

        # Check if the custom tag is present in the Exif data
        if custom_tag not in exifdata:
            # Custom tag doesn't exist, add it with an initial value
            exifdata[custom_tag] = ''.encode('utf-16')

        # Check if the custom tag is present in the Exif data
        if custom_tag in exifdata:
            # Decode the current tags string and remove null characters
            tags_string_concat = exifdata[custom_tag].decode('utf-16').replace('\x00', '').replace(', ',',').replace(' ,',',')

            # Split the tags into a list
            tags_list = [tag.strip() for tag in re.split(r'[;,]', tags_string_concat)]
            #tags_list = list(set(tag.strip() for tag in re.split(r'[;,]', tags_string_concat)))
            #tags_list = tags_string_concat.split(',')

            # Convert single tag to a list
            if isinstance(tags, str):
                tags = [tags]
            #elif ',' in tags:
            #    tags = tags.split(',')

            tags_list = list(set(tags_list))

            if command == 'add':
                # Add the tags if not present
                for tag in tags:
                    if tag in tags_list:
                        print(tag + " Already present")
                    if tag not in tags_list:
                        print("Need to add " + tag)
                        tags_list.append(tag)
            elif command == 'remove':
                # Remove the tags if present
                for tag in tags:
                    if tag in tags_list:
                        print("Need to remove " + tag)
                        tags_list.remove(tag)
            elif command == 'show':
                # Return the list of tags or None if empty
                print(f"Exif tags {command}ed successfully.")
                return tags_list if tags_list else None
            elif command == 'update':
                # Update an existing tag with a new value
                if new_value is not None:
                    for tag in tags:
                        if tag in tags_list:
                            index = tags_list.index(tag)
                            print("updating tag " +  tag + " from " + index + " to " + new_value)
                            tags_list[index] = new_value
                        else:
                            print(f"Tag '{tag}' not found for updating.")
                else:
                    print("Missing new_value for 'update' command.")
                    return
            elif command == 'clear':
                # Clear all tags
                tags_list = []
            elif command == 'count':
                # Get the count of tags
                print(f"Exif tags {command} completed successfully.")
                return len(tags_list)
            elif command == 'search':
                # Check if a specific tag exists
                print(f"Exif tags {command}ed successfully.")
                return any(tag in tags_list for tag in tags)
            else:
                print("Invalid command. Please use 'add', 'remove', 'show', 'update', 'clear', 'count', or 'search'.")
                return


            # Check if the tags have changed
            new_tags_set = set(tags_list)
            if set(tags) - new_tags_set:
            #if updated_tags_string != tags_string_concat:
                # Encode the modified tags string and update the Exif data
                # Join the modified tags list into a string
                updated_tags_string = ','.join(tags_list)

                exifdata[custom_tag] = updated_tags_string.encode('utf-16')

                # Save the image with updated Exif data
                image.save(filename, exif=exifdata)
                print(f"Exif tags {command}ed successfully.")
                os.utime(filename, (original_atime, original_mtime))
                print(f"atime and mtime restored.")

            else:
                print("No changes in tags. File not updated.")

        else:
            print("Custom tag not found in Exif data.")
    else:
        print(f"File not found: {filename}")

def load_image_in_thread(image_path):

    try:
        image1 = Image.open(image_path).resize((400, 300), Image.LANCZOS)
    except Exception as e:
        print(f'{e}')
        prepend_string_to_filename(image_path,'corrupt_')
        return None

    return ImageTk.PhotoImage(image1)


################################################################################################
#GUI
################################################################################################

class ImageTextDisplay:
    def __init__(self, root):
        self.root = root
        self.image_text_list = []
        self.current_index = 0
        self.next_index = 0  # Index for the next image
        self.text_str = ""
        self.auto = False
        self.recursive_var = tk.BooleanVar(value=True)  # Variable to track the state of the recursive checkbox

        self.create_widgets()
        self.update_file_list()

    def create_widgets(self):
        # Entry for image directory
        self.image_directory_entry = tk.Entry(self.root, width=50)
        self.image_directory_entry.insert(0, defaultdir)  # Default directory
        self.image_directory_entry.pack()

        # Recursive checkbox
        self.recursive_checkbox = tk.Checkbutton(self.root, text="Recursive", variable=self.recursive_var)
        self.recursive_checkbox.pack()

        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # Text display
        self.text_var = tk.StringVar()
        self.text_label = tk.Label(self.root, textvariable=self.text_var)
        self.text_label.pack()

        # Next button
        self.next_button = tk.Button(self.root, text="Next", command=self.show_next)
        self.next_button.pack()

        # Update button
        self.update_button = tk.Button(self.root, text="Update directory", command=self.update_file_list)
        self.update_button.pack()

        # Update button
        self.apply_button = tk.Button(self.root, text="apply interrogation", command=self.apply_interrogation)
        self.apply_button.pack()

        # Auto Play button
        self.auto_play_button = tk.Button(self.root, text="Auto Play", command=self.auto_play)
        self.auto_play_button.pack()

        # Initial display
        self.show_current()

    def inference(ci, image, mode):
        image = image.convert('RGB')
        if mode == 'best':
            return ci.interrogate(image)
        elif mode == 'classic':
            return ci.interrogate_classic(image)
        else:
            return ci.interrogate_fast(image)
    
    def apply_interrogation(self):
        if self.image_text_list:
            image_path, text_set = self.image_text_list[self.current_index]
            test = self.text_str.replace(', ',',').replace(' ,',',')
            test = test.split(',')
            print("hi")
            print(image_path)
            print(str(test))
            modify_exif_tags(image_path, test, 'add')

    def auto_play(self):
        self.auto = True
        # Auto play through all files
        for _ in range(len(self.image_text_list)):
            self.show_next()
            #self.root.update_idletasks()  # Update the Tkinter GUI
            #self.apply_interrogation(self)

            # Load the next image in a separate thread
            #next_image_path, _ = self.image_text_list[self.next_index]
            #thread = threading.Thread(target=self.load_next_image, args=(next_image_path,))
            #thread = threading.Thread(target=load_image_in_thread, args=(next_image_path,))
            #thread.start()

            #self.root.after(2000)  # Wait for 2000 milliseconds (2 seconds) before showing the next image
        self.auto = False

    #def load_next_image(self, image_path):
        #global photo
    #    photo = load_image_in_thread(image_path)
    #    self.image_label.config(image=photo)
    #    self.image_label.image = photo
        
    def show_current(self):
        if self.image_text_list:
            image_path, text_set = self.image_text_list[self.current_index]

            print(f'{image_path} {text_set}')
            photo = load_image_in_thread(image_path)
            #image1 = Image.open(image_path).resize((400, 300), Image.LANCZOS)
            #global photo
            #photo = ImageTk.PhotoImage(photo)
            if photo is not None:
                self.image_label.config(image=photo)
                self.image_label.image = photo
                #self.after(1000,self.)

                #self.root.update_idletasks()  # Update the Tkinter GUI
                result = image_to_wd14_tags(image_path)
                if result is not None:
                    text_set = result[1]

                    # Display text
                    text_list = list(text_set)
                    self.text_str = ", ".join(text_list)
                    self.text_var.set(self.text_str)
                    if self.auto == True:
                        self.apply_interrogation()
            else:
                print("image was corrupt")
            # Preload the next image in the background
            self.root.update_idletasks()  # Update the Tkinter GUI
            #self.preload_next_image()

        else:
            self.text_var.set("No images found")

    def show_next(self):
        if self.image_text_list:
            self.current_index = (self.current_index + 1) % len(self.image_text_list)
            self.show_current()

    def update_file_list(self):
        # Get the image directory from the entry widget
        image_directory = self.image_directory_entry.get()

        if self.recursive_var.get():
            image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_directory) for f in filenames]
        else:
            image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]


        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg'))]

        # Filter files based on image extensions
        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg'))]

        # List all files in the directory
        #image_files = [f for f in os.listdir(image_files) if f.lower().endswith(('.png', '.jpg', '.jpeg',))]

        # Create image_text_list dynamically
        #self.image_text_list = [(os.path.join(image_files, file), set()) for file in image_files]
        self.image_text_list = [(file_path, set()) for file_path in image_files]

        # Update the display
        self.show_current()

gui = True
defaultdir = '/folder/to/process'

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



RE_SPECIAL = re.compile(r'([\\()])')

if gui == True:
    #photo = None
    root = tk.Tk()
    root.title("Image Text Display")

    # Creating an instance of ImageTextDisplay
    app = ImageTextDisplay(root)

    # Start the main event loop
    root.mainloop()

else:
    modelarray = {
                #'ViT-L-14': 'ViT-L-14/openai',
                #'ViT-L-14': 'immich-app/ViT-L-14__openai',
                #'ViT-H-14': 'ViT-H-14/laion2b_s32b_b79',
       #         'ViT-H-14': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
                #'wd14': 'wd14-convnext',
                'wd14': 'saltacc/wd-1-4-anime',
                'blip-base': 'Salesforce/blip-image-captioning-base',   # 990MB
       #         'blip-large': 'Salesforce/blip-image-captioning-large', # 1.9GB
            #    'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',              # 15.5GB
            #    'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',      # 15.77GB
                'git-large-coco': 'microsoft/git-large-coco'           # 1.58GB
                }

    for root, dirs, files in os.walk(defaultdir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg')):
                try:
                    fullpath = os.path.join(root,filename)
                    
                    for each,desc in modelarray.items():

                        print("using: " + each)

                        processor = BlipProcessor.from_pretrained(desc)
                        model = BlipForConditionalGeneration.from_pretrained(desc)

                        image = Image.open(fullpath).convert('RGB')

                        inputs = processor(image, return_tensors="pt")

                        out = model.generate(**inputs)
                        print(f"{fullpath}. {each} {processor.decode(out[0], skip_special_tokens=True)}")
                    
                        print("press a key to continue")
                        input()
                    break

                    result2 = result[1]

                    
                    print("hi")
                    print(fullpath)
                    print(str(result))
                    if result2 is not None:
                        tagname = 'EXIF:XPKeywords'
                        #modify_exif_tags(fullpath, result2, 'add',tagname)
                except Exception as e:
                    print(f"oops.  {e}")



