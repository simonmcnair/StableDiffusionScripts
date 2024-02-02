__author__='Simon McNair'

#python3.11 -m venv venv
#source ./venv/bin/activate

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
import numpy
import nltk
from PIL.ExifTags import TAGS
from PIL import Image, ImageTk
import platform

from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession
import pandas as pd
import cv2
import numpy as np
from typing import Mapping, Tuple, Dict
import re
#import threading
from pathlib import Path

from nltk.corpus import wordnet
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.applications.imagenet_utils import decode_predictions,preprocess_input

import tkinter as tk


def get_script_name():
    # Use os.path.basename to get the base name (script name) from the full path
    #basename = os.path.basename(path)
    return Path(__file__).stem
    #return os.path.basename(__file__)

def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))

def get_operating_system():
    system = platform.system()
    return system

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

def image_make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

# noinspection PyUnresolvedReferences
def image_smart_resize(img, size):
    # Assumes the image has already gone through image_make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    else:  # just do nothing
        pass

    return img

################################################################################################
################################################################################################

class CLIPInterrogator:
    def __init__(
            self,
            repo='SmilingWolf/wd-v1-4-vit-tagger-v2',
            model_path='model.onnx',
            tags_path='selected_tags.csv',
            mode: str = "auto"
    ) -> None:
        self.__repo = repo
        self.__model_path = model_path
        self.__tags_path = tags_path
        self._provider_mode = mode

        self.__initialized = False
        self._model, self._tags = None, None

    def _init(self) -> None:
        if self.__initialized:
            return

        model_path = hf_hub_download(self.__repo, filename=self.__model_path)
        tags_path = hf_hub_download(self.__repo, filename=self.__tags_path)

        self._model = InferenceSession(str(model_path))
        self._tags = pd.read_csv(tags_path)

        self.__initialized = True

    def _calculation(self, image: Image.Image)  -> pd.DataFrame:
        self._init()

        _, height, _, _ = self._model.get_inputs()[0].shape

        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = image_make_square(image, height)
        image = image_smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        # evaluate model
        input_name = self._model.get_inputs()[0].name
        label_name = self._model.get_outputs()[0].name
        confidence = self._model.run([label_name], {input_name: image})[0]

        full_tags = self._tags[['name', 'category']].copy()
        full_tags['confidence'] = confidence[0]

        return full_tags

    def interrogate(self, image: Image) -> Tuple[Dict[str, float], Dict[str, float]]:
        full_tags = self._calculation(image)

        # first 4 items are for rating (general, sensitive, questionable, explicit)
        ratings = dict(full_tags[full_tags['category'] == 9][['name', 'confidence']].values)

        # rest are regular tags
        tags = dict(full_tags[full_tags['category'] != 9][['name', 'confidence']].values)

        return ratings, tags
################################################################################################
################################################################################################

################################################################################################
def generate_tags(prediction):
    #variable
    tags=''
    #filtering predicitions with 60% or more probability
    #tags along with their synonyms are returned
    for x in range(0,5):
        if float(prediction[0][x][2])*100>=50:
            #getting synonyms
            if len(tags)==0:
                tags=str(prediction[0][x][1]) + ',' + get_synonyms(str(prediction[0][x][1]))
            else:
                tags = tags + ', '+ str(prediction[0][x][1]) + ',' + get_synonyms(str(prediction[0][x][1]))

    #returning tags
    return tags
################################################################################################
def get_synonyms(word):
    synonyms=''
    for syn in wordnet.synsets(word):
        for lma in syn.lemmas():
            if not str(lma.name()).lower()==str(word).lower():
                if len(synonyms)==0:
                    synonyms=synonyms + lma.name()
                else:
                    synonyms = synonyms + ', ' + lma.name()

    return synonyms
################################################################################################
def get_predictions(model,imagepath):
    #loading image
    #converting to array and preprocessing image
    img=image.load_img(path=os.path.abspath(imagepath),target_size=(224,224))
    img_arr=image.img_to_array(img)
    img_arr=numpy.expand_dims(img_arr,axis=0)
    img_arr=preprocess_input(img_arr)
    #prediciton
    pred=model.predict(img_arr)
    return decode_predictions(pred)
################################################################################################
def different_tag_process(imagefile):
    model = ResNet50(weights='imagenet')
    prediction=get_predictions(model,imagefile)
    #generating tags from prediction
    tags = generate_tags(prediction)

    return prediction, tags
################################################################################################

################################################################################################
#def image_to_wd14_tags(filename, image:Image.Image) \
def image_to_wd14_tags(filename,modeltouse='wd14-vit-v2') \
        -> Tuple[Mapping[str, float], str, Mapping[str, float]]:
    
    try:
        image = Image.open(filename)
        print("image: " + filename + " successfully opened.  Continue processing ")
    except Exception as e:
        print("Processfile Exception1: " + " failed to open image : " + filename + ". FAILED Error: " + str(e) + ".  Skipping")
        return None

    try:
        model = CLIPInterrogatorModels[modeltouse]
        ratings, tags = model.interrogate(image)

        filtered_tags = {
            tag: score for tag, score in tags.items()
            #if score >= .35
            if score >= .80
        }

        text_items = []
        tags_pairs = filtered_tags.items()
        tags_pairs = sorted(tags_pairs, key=lambda x: (-x[1], x[0]))
        for tag, score in tags_pairs:
            tag_outformat = tag
            tag_outformat = tag_outformat.replace('_', ' ')
            tag_outformat = re.sub(RE_SPECIAL, r'\\\1', tag_outformat)
            text_items.append(tag_outformat)
        #output_text = ', '.join(text_items)
        #return ratings, output_text, filtered_tags
        return ratings, text_items, filtered_tags
    except Exception as e:
        print(f"Exception getting tags from image {filename}.  Error: {e}" )
        return None
################################################################################################

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
    #api_key = apikey
    #print("API Key:", api_key)
    print("local override file is " + localoverridesfile)

else:
    print("local override file would be " + localoverridesfile)




CLIPInterrogatorModels: Mapping[str, CLIPInterrogator] = {
    'wd14-vit-v2': CLIPInterrogator(),
    'wd14-convnext': CLIPInterrogator(repo='SmilingWolf/wd-v1-4-convnext-tagger'),
    'ViT-L-14/openai': CLIPInterrogator(),
    'ViT-H-14/laion2b_s32b_b79': CLIPInterrogator(),
    'ViT-L-14/openai': CLIPInterrogator()
}

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
    modelarray = [
        'ViT-L-14/openai',
        'ViT-H-14/laion2b_s32b_b79',
        'ViT-L-14/openai',  # This line had a missing comma in the original list
        'wd14-convnext'
#       'blip-base': 'Salesforce/blip-image-captioning-base',   # 990MB
#       'blip-large': 'Salesforce/blip-image-captioning-large', # 1.9GB
#       'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',              # 15.5GB
#       'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',      # 15.77GB
#       'git-large-coco': 'microsoft/git-large-coco',           # 1.58GB

    ]

    for root, dirs, files in os.walk(defaultdir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg')):
                try:
                    fullpath = os.path.join(root,filename)
                    
                    for each in modelarray:
                        print(f"using {each} to process {fullpath}")
                        ratings, text_items, filtered_tags = image_to_wd14_tags(fullpath,each)
                        print(f"using {each} caption returned was {ratings}. {text_items} {filtered_tags}")

                    prediction, tags = different_tag_process(fullpath)
                    print(f"using resnet50 caption returned was {prediction}  . {tags}")

                    ratings, tags = CLIPInterrogator.interrogate()
                    

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



