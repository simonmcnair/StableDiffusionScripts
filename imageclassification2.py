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

import pandas as pd
import cv2
import numpy as np
from typing import Mapping, Tuple, Dict
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession

#importing libraries
import os
#import glob

from PIL.ExifTags import TAGS
from PIL import Image, ImageTk
#bewlow for pngs
from PIL import PngImagePlugin, Image
#pip install piexif
#import piexif
#import piexif.helper

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


def is_person(word):
    import spacy

    # Load the spaCy model for English
    nlp = spacy.load("en_core_web_sm")

    # Process the text with spaCy
    doc = nlp(word)

    # Check if any entity in the text is a person
    is_person = any(entity.label_ == 'PERSON' for entity in doc.ents)

    if is_person:
       return True
    else:
       return False

def ddb(imagefile):

    #os.environ["CUDA_VISIBLE_DEVICES"]=""

    import torch
    # Load the model

    model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')
    model.eval()


    from torchvision import transforms
    input_image = Image.open(imagefile) # load an image of your choice
    preprocess = transforms.Compose([
        transforms.Resize(360),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7137, 0.6628, 0.6519], std=[0.2970, 0.3017, 0.2979]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    import json
    import urllib, urllib.request

    with urllib.request.urlopen("https://github.com/RF5/danbooru-pretrained/raw/master/config/class_names_6000.json") as url:
        class_names = json.loads(url.read().decode())

    # The output has unnormalized scores. To get probabilities, you can run a sigmoid on it.
    probs =  torch.sigmoid(output[0]) # Tensor of shape 6000, with confidence scores over Danbooru's top 6000 tags
    thresh=0.2
    tmp = probs[probs > thresh]
    inds = probs.argsort(descending=True)
    txt = 'Predictions with probabilities above ' + str(thresh) + ':\n'
    for i in inds[0:len(tmp)]:
        txt += class_names[i] + ': {:.4f} \n'.format(probs[i].cpu().numpy())
    #plt.text(input_image.size[0]*1.05, input_image.size[1]*0.85, txt)
    return txt



def unumcloud(image):
    from transformers import AutoModel, AutoProcessor
    import torch
    model = AutoModel.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)

    #prompt = "Question or Instruction"
    prompt = ""
    image = Image.open(image)

    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=256,
            eos_token_id=151645,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]

def prepend_string_to_filename(fullpath, prefix):
    # Split the full path into directory and filename
    directory, filename = os.path.split(fullpath)

    # Prepend the prefix to the filename
    new_filename = f"{prefix}{filename}"

    # Join the directory and the new filename to get the updated full path
    new_fullpath = os.path.join(directory, new_filename)

    return new_fullpath

def nlpconnect(fileinput):

    from transformers import pipeline

    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

    return image_to_text(fileinput)

def blip2_opt_2_7b(inputfile):

    # pip install accelerate bitsandbytes
    # pip install -q -U bitsandbytes
    # pip install -q -U git+https://github.com/huggingface/transformers.git
    # pip install -q -U git+https://github.com/huggingface/peft.git
    # pip install -q -U git+https://github.com/huggingface/accelerate.git
    import torch
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from transformers import AutoModelForCausalLM
    from transformers import BitsAndBytesConfig


    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    #model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_4bit=True, device_map="auto")
    model_nf4 = AutoModelForCausalLM.from_pretrained("Salesforce/blip2-opt-2.7b", quantization_config=nf4_config)

    raw_image = Image.open(inputfile).convert('RGB')

    #question = "how many dogs are in the picture?"
    question = ""
    inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True).strip())

def blip_large(imagepath,model='small'):

    from transformers import BlipProcessor, BlipForConditionalGeneration

    if model == 'small':
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
    elif model == 'large':
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

    raw_image = Image.open(imagepath).convert('RGB')

    # conditional image captioning
    text = ""
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)

    res = processor.decode(out[0], skip_special_tokens=True)
    return res


def write_pnginfo(filename,tags):
    if os.path.exists(filename):
        writefile = False
        image = Image.open(filename)
        metadata = PngImagePlugin.PngInfo()
        inferencefound = False
        for key, value in image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                if key == 'exif':
                    print("exif data breaks the file.  Skip {filename}")
                    continue
                elif key == 'parameters':
                    print(f"Stable Diffusion file. {filename}: {value}")
                    metadata.add_text(key, value)
                    sd = True
                elif key =='Inference':
                    print(f"inference text already exists. {filename}: {value}")
                    inferencefound = True
                    metadata.add_text(key,value)
                else:
                    print(f"Other: {key}.  {value}")
                    metadata.add_text(key, value)
        if inferencefound == False:
            metadata.add_text('Inference',(';'.join(tags)))
            writefile = True

        if writefile == True:
            original_mtime = os.path.getmtime(filename)
            original_atime = os.path.getatime(filename)
            try:
                image.save(filename,format="PNG",pnginfo=metadata)
            except Exception as e:
                print(f"error {e}")
            os.utime(filename, (original_atime, original_mtime))
            print(f"atime and mtime restored.")
        

def modify_exif_tags(filename, tags, command, new_value=None, tagname= None):
    # Check if the file exists
    tags_list = []
    does_image_have_tags = False
    if os.path.exists(filename):
        # Open the image

        original_mtime = os.path.getmtime(filename)
        original_atime = os.path.getatime(filename)

        image = Image.open(filename)

        # Get the Exif data
        exifdata = image.getexif()

    #    exif_dict = piexif.load(filename)

        # Print the Exif data
    #    for ifd, data in exif_dict.items():
    #        print(f"IFD {ifd}:")
    #        for tag, value in data.items():
    #            tag_name = piexif.TAGS[ifd][tag]["name"]
               # if tag_name == 'XPKeywords':

               #     keywordsraw = exif_dict["0th"][piexif.ImageIFD.XPKeywords]
                    #keywordsraw = str(bytes(keywordsraw), "utf-16le") 
               #     test =  str(bytes(value), "utf-16le") 
               #     print(test)                 
                  #  tags = ""
                  #  for num in exif_dict["0th"][piexif.ImageIFD.XPKeywords]:
                  #      tags += chr(num)

                  #  value = xp_keywords_string
     #           print(f"  {tag_name}: {value}")

        # Convert single tag to a list
        if isinstance(tags, str):
            tags = [tags]

        if exifdata == None:
            print("No exifdata")
            found = False
        else:
            # Use a custom tag (you can modify this based on your requirements)
            found = False
            if tagname is not None:
                    for pil_tag, pil_tag_name in TAGS.items():
                        if pil_tag_name == tagname:
                            #custom_tag = hex(pil_tag_name)
                            custom_tag = pil_tag
                            print(f"using {pil_tag} for {tagname} tag")
                            found = True
                            break
        if found == False or tagname == None:
            # 40094:0x9C9E:'XPKeywords'
            print("No exifdata or tagname = None.  Using XPKeywords for tag")
            #custom_tag = 0x9C9E
            custom_tag = 40094

        # Check if the custom tag is present in the Exif data
        if custom_tag not in exifdata:
            # Custom tag doesn't exist, add it with an initial value
            
            exifdata[custom_tag] = ''.encode('utf-16le')
            #exifdata[custom_tag] = ''.encode('utf-16')
            print("image doesn't currently have any tags")
            current_tags = []
        else:
            does_image_have_tags = True
            print("image currently has tags")

            # Decode the current tags string and remove null characters
            current_tags = exifdata[custom_tag].decode('utf-16le').replace('\x00', '').replace(', ',',').replace(' ,',',')
            #current_tags = exifdata[custom_tag].decode('utf-16').replace('\x00', '').replace(', ',',').replace(' ,',',')

            # Split the tags into a list
            current_tags = [current_tags.strip() for current_tags in re.split(r'[;,]', current_tags)]
            #tags_list = list(set(tag.strip() for tag in re.split(r'[;,]', tags_string_concat)))
            #tags_list = tags_string_concat.split(',')

            #remove any dupes
            current_tags = list(set(current_tags))
            #remove any empty values
            current_tags = {value for value in current_tags if value}

            if len(current_tags) == 0:
                print("current_tags is there, but has no tags in")

        if command == 'add':
            # Add the tags if not present
            if does_image_have_tags:
                tags_to_add = set(tags) - set(current_tags)
                tags_list.extend(tags_to_add)
            else:
                tags_list.extend(tags)

        elif command == 'remove':
            if does_image_have_tags:
                tags_to_remove = set(tags) & set(current_tags)
                tags_list = list(set(tags_list) - tags_to_remove)
            else:
                # If does_image_have_tags is False, you can decide if there's a specific removal logic
                print("does_image_have_tags is False, skipping removal.")

        elif command == 'show':
            # Return the list of tags or None if empty
            print(f"Exif tags {command}ed successfully.")
            return tags_list if tags_list else None
        elif command == 'update':
            # Update an existing tag with a new value
                if new_value is not None:
                    if does_image_have_tags:
                        tags_to_add = set(tags) - set(current_tags)
                        tags_to_remove = set(current_tags) & set(tags)

                        tags_set = (set(tags_list) - tags_to_remove) | tags_to_add
                        tags_list = list(tags_set)
                    else:
                        # If does_image_have_tags is False, you can decide if there's a specific update logic
                        print("does_image_have_tags is False, skipping update.")                            
                else:
                    print("Missing new_value for 'update' command.")
                    return
        elif command == 'clear':
            # Clear all tags
            tags_list = []
        elif command == 'count':
            # Get the count of tags
            print(f"Exif tags {command} completed successfully.")
            if does_image_have_tags == True:
                return len(tags_list)
            else:
                return 0
        elif command == 'search':
            # Check if a specific tag exists
            if does_image_have_tags == True:
                print(f"Exif tags {command}ed successfully.")
                return any(tag in current_tags for tag in tags)
            else:
                return ''
        else:
            print("Invalid command. Please use 'add', 'remove', 'show', 'update', 'clear', 'count', or 'search'.")
            return

        # Check if the tags have changed
        if does_image_have_tags == True:
            #remove dupes
            new_tags_set = set(tags_list)
            #remove empty/null
            new_tags_set = {value for value in new_tags_set if value}

        if does_image_have_tags == False or len(tags_list) > 0:
            if does_image_have_tags == False:
                print(f"no tags originally.  Need to add tags {str(list(tags_list))}.")
            else:
                print(f"need to add tags {str(list(tags_list))}.  Current tags are {str(list(current_tags))}")

        #if updated_tags_string != tags_string_concat:
            # Encode the modified tags string and update the Exif data
            # Join the modified tags list into a string
            updated_tags_string = ';'.join(tags_list)

            #exifdata[custom_tag] = updated_tags_string.encode('utf-16')
            exifdata[custom_tag] = updated_tags_string.encode('utf-16le')

            # Save the image with updated Exif data
            image.save(filename, exif=exifdata)
            print(f"Exif tags {command}ed successfully to {filename}.")
            os.utime(filename, (original_atime, original_mtime))
            print(f"atime and mtime restored.")
        else:
            print(f"No changes in tags for file {filename}. File not updated.")
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

def image_smart_resize(img, size):
    # Assumes the image has already gone through image_make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    else:  # just do nothing
        pass

    return img

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
        print(f"model path is {model_path}")

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

CLIPInterrogatorModels: Mapping[str, CLIPInterrogator] = {
    'wd14-vit-v2': CLIPInterrogator(),
    'wd14-convnext': CLIPInterrogator(repo='SmilingWolf/wd-v1-4-convnext-tagger'),
    'ViT-L-14/openai': CLIPInterrogator(),
    'ViT-H-14/laion2b_s32b_b79': CLIPInterrogator(),
    'ViT-L-14/openai': CLIPInterrogator(),
    'wd-v1-4-moat-tagger-v2': CLIPInterrogator(repo='SmilingWolf/wd-v1-4-moat-tagger-v2'),
    'wd-v1-4-swinv2-tagger-v2': CLIPInterrogator(repo='SmilingWolf/wd-v1-4-swinv2-tagger-v2'),
    'wd-v1-4-convnext-tagger-v2': CLIPInterrogator(repo='SmilingWolf/wd-v1-4-convnext-tagger-v2'),
    'wd-v1-4-convnextv2-tagger-v2': CLIPInterrogator(repo='SmilingWolf/wd-v1-4-convnextv2-tagger-v2'),
    'wd-v1-4-vit-tagger-v2': CLIPInterrogator(repo='SmilingWolf/wd-v1-4-vit-tagger-v2')
}

def image_to_wd14_tags(filename,modeltouse='wd14-vit-v2') \
        -> Tuple[Mapping[str, float], str, Mapping[str, float]]:
    
    try:
        image = Image.open(filename)
        print("image: " + filename + " successfully opened.  Continue processing ")
    except Exception as e:
        print("Processfile Exception1: " + " failed to open image : " + filename + ". FAILED Error: " + str(e) + ".  Skipping")
        return None

    try:
        print(modeltouse)
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
        #        'wd14': 'saltacc/wd-1-4-anime',
                'wd' : 'SmilingWolf/wd-v1-4-vit-tagger-v2',
       #         'blip-base': 'Salesforce/blip-image-captioning-base',   # 990MB
       #         'blip-large': 'Salesforce/blip-image-captioning-large', # 1.9GB
            #    'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',              # 15.5GB
            #    'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',      # 15.77GB
        #        'git-large-coco': 'microsoft/git-large-coco'           # 1.58GB
                }

    for root, dirs, files in os.walk(defaultdir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg','.png')):
                try:
                    fullpath = os.path.join(root,filename)
                    
                    # for each,desc in modelarray.items():

                    #     print("using: " + each)

                    #     processor = BlipProcessor.from_pretrained(desc)
                    #     model = BlipForConditionalGeneration.from_pretrained(desc)

                    #     image = Image.open(fullpath).convert('RGB')

                    #     inputs = processor(image, return_tensors="pt")

                    #     out = model.generate(**inputs)
                    #     print(f"{fullpath}. {each} {processor.decode(out[0], skip_special_tokens=True)}")
                    
                    #     print("press a key to continue")
                    #     input()
                    # break

                    #result = ddb(fullpath)
                    
                    #result = image_to_wd14_tags(fullpath,'wd14-vit-v2')
                    #print(f"{fullpath} . {str(result)} . wd14-vit-v2") 
                    #result = image_to_wd14_tags(fullpath,'wd14-convnext')
                    #print(f"{fullpath} . {str(result)} . wd14-convnext")#377MB model.onnx
                    #result = image_to_wd14_tags(fullpath,'wd-v1-4-moat-tagger-v2')
                    #print(f"{fullpath} . {str(result)} . wd-v1-4-moat-tagger-v2")#377MB model.onnx
                    #result = image_to_wd14_tags(fullpath,'wd-v1-4-swinv2-tagger-v2')
                    #print(f"{fullpath} . {str(result)} . wd-v1-4-swinv2-tagger-v2")#377MB model.onnx
                    #result = image_to_wd14_tags(fullpath,'wd-v1-4-convnext-tagger-v2')
                    #print(f"{fullpath} . {str(result)} . wd-v1-4-convnext-tagger-v2")#377MB model.onnx
                    result = image_to_wd14_tags(fullpath,'wd-v1-4-convnextv2-tagger-v2')
                    print(f"{fullpath} . {str(result)} . wd-v1-4-convnextv2-tagger-v2")#377MB model.onnx
                    #result = image_to_wd14_tags(fullpath,'wd-v1-4-vit-tagger-v2')
                    #print(f"{fullpath} . {str(result)} . wd-v1-4-vit-tagger-v2")#377MB model.onnx

                    #result = image_to_wd14_tags(fullpath,'ViT-L-14/openai')
                    #print(f"{fullpath} . {str(result)} . ViT-L-14/openai")
                    #result = image_to_wd14_tags(fullpath,'ViT-H-14/laion2b_s32b_b79')
                    #print(f"{fullpath} . {str(result)} . ViT-H-14/laion2b_s32b_b79")
                    #result = image_to_wd14_tags(fullpath,'ViT-L-14/openai')
                    #print(f"{fullpath} . {str(result)} . ViT-L-14/openai")

                    #test = blip2_opt_2_7b(fullpath)
                    #test = blip_large(fullpath)
                    #test = unumcloud(fullpath)
                    #test = nlpconnect(fullpath)
                    #print(f"{fullpath} . {str(test)}")
                    #exit()

                    if result is not None:
                        result2 = result[1]

                        if len(result2) > 0:
                            print(str(result2))
                            tagname = 'XPKeywords'
                            #tagname = 'EXIF:XPKeywords'
                            if filename.lower().endswith(('.png')):
                                write_pnginfo(fullpath, result2)
                            elif filename.lower().endswith(('.jpg')):
                                modify_exif_tags(fullpath, result2, 'add',None,tagname)
                        else:
                            print("stuff detected but not relevant/length 0")
                    else:
                        print(f"nothing detected for {fullpath}.  Odd")
                except Exception as e:
                    print(f"oops.  {e}")