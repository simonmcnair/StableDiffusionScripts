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
import utils
import time
#pip install numpy
#pip install huggingface_hub
#pip install onnxruntime
#pip install pandas
#pip install opencv
#pip install opencv-python
#pip install keras
#pip install tensorflow
#sudo apt-get install python3-tk
import torch

import pandas as pd
import cv2
import numpy as np
from typing import Mapping, Tuple, Dict
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession

from exiftool import ExifToolHelper
#pip install pyexiftool

#importing libraries
import os
import re

#import glob

from PIL.ExifTags import TAGS
from PIL import Image, ImageTk
#below for pngs
from PIL import PngImagePlugin, Image
#.\venv\Scripts\pip.exe install pillow
#pip install piexif
#import piexif
#import piexif.helper

import clip_interrogator
from clip_interrogator import Config, Interrogator, list_clip_models
import platform
from transformers import BlipProcessor, BlipForConditionalGeneration

import re
#import threading
from pathlib import Path

import tkinter as tk

def timing_decorator(func):
    global timing_debug

    def wrapper(*args, **kwargs):
        if timing_debug == False:
            return
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' took {end_time - start_time} seconds to execute.")
        return result
    return wrapper

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

@timing_decorator
def is_person(snippet):

    try:
        for sent in nltk.sent_tokenize(snippet):
            tokens = nltk.tokenize.word_tokenize(sent)
            tags = st.tag(tokens)
            for tag in tags:
                if tag[1]=='PERSON': 
                    logger.info(f"{tag[0]} is a persons name !!!!!!")
                    return True
                #elif tag[1]=='ORGANIZATION':
                #    logger.info(f"{tag} is probably a persons name !!!!!!{tag[1]}")
                #    return True  
        return
    except Exception as e:
        logger.error(f"Error in isperson{e}")

@timing_decorator
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            import gc
            #logger.info("mem before")
            #logger.info(torch.cuda.memory_summary(device=None, abbreviated=False))
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            #del variables

            #logger.info("mem after")
            #logger.info(torch.cuda.memory_summary(device=None, abbreviated=False))

@timing_decorator
def low_vram():
    if torch.cuda.is_available():
        vram_total_mb = torch.cuda.get_device_properties('cuda').total_memory / (1024**2)
        vram_info = f"GPU VRAM: **{vram_total_mb:.2f}MB**"
        if vram_total_mb< 8:
            vram_info += "<br>Using low VRAM configuration"
            logger.info(f"{vram_info}")
    if vram_total_mb <= '4': return False 
    return True

@timing_decorator
def return_vram():
    if torch.cuda.is_available():
        vram_total_mb = torch.cuda.get_device_properties('cuda').total_memory / (1024**2)
    return vram_total_mb


@timing_decorator
def load(clip_model_name):
    global ci
    if ci is None:
        logger.info(f"Loading CLIP Interrogator {clip_interrogator.__version__}...")

        config = Config(
            cache_path = 'models/clip-interrogator',
            clip_model_name=clip_model_name,
        )

        if low_vram:
            logger.info("low vram")
            config.apply_low_vram_defaults()
            config.chunk_size = 512
        ci = Interrogator(config)

    if clip_model_name != ci.config.clip_model_name:


        ci.config.clip_model_name = clip_model_name
        torch_gc()
        #with Timer() as modelloadtime:
        ci.load_clip_model()
        #logger.info(f"loading model took {modelloadtime.last} to load")
        #return res, modelloadtime.last

@timing_decorator
def ddb(imagefile):

    #os.environ["CUDA_VISIBLE_DEVICES"]=""

    #blip_large
    import torch
    from torchvision import transforms
    import json
    import urllib, urllib.request

    #Blip_large
    from transformers import BlipProcessor, BlipForConditionalGeneration


    # Load the model

    model = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')
    model.eval()


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



@timing_decorator
def unumcloud(image):

    #unumcloud(image):
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

@timing_decorator
def prepend_string_to_filename(fullpath, prefix):
    # Split the full path into directory and filename
    directory, filename = os.path.split(fullpath)

    # Prepend the prefix to the filename
    new_filename = f"{prefix}{filename}"

    # Join the directory and the new filename to get the updated full path
    new_fullpath = os.path.join(directory, new_filename)

    return new_fullpath

@timing_decorator
def nlpconnect(fileinput):
    #nlpconnect
    from transformers import pipeline

    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    return image_to_text(fileinput)

@timing_decorator
def blip2_opt_2_7b(inputfile):
    #blip2_opt_2_7b
    import torch
    from PIL import Image
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from transformers import AutoModelForCausalLM
    from transformers import BitsAndBytesConfig
    # pip install accelerate bitsandbytes
    # pip install -q -U bitsandbytes
    # pip install -q -U git+https://github.com/huggingface/transformers.git
    # pip install -q -U git+https://github.com/huggingface/peft.git
    # pip install -q -U git+https://github.com/huggingface/accelerate.git



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
    logger.info(processor.decode(out[0], skip_special_tokens=True).strip())

@timing_decorator
def use_GPU_interrogation(image_path,model_name="ViT-L-14/openai"):
    
    load("ViT-L-14/openai")
        #models = list_clip_models()
    #logger.info(f"supported models are {models}")
    logger.info("load image")
    image = Image.open(image_path).convert('RGB')
    logger.info("convert RGB")
    #ci = Interrogator(Config(clip_model_name=model_name))
    #logger.info("create CI")
    res = ci.interrogate(image)
    logger.info ("interrogation complete")
    logger.info(res)
    return (res)


@timing_decorator
def get_description_keywords_tag(filetoproc, istagged=False):
    res = {}
    taglist =[]
    taglist.append("IPTC:Keywords")#list
    taglist.append("XMP:TagsList")
    taglist.append("XMP:Hierarchicalsubject")
    #taglist.append("XMP:Categories")
    taglist.append("XMP:CatalogSets")
    taglist.append("XMP:LastKeywordXMP")
    taglist.append("EXIF:XPKeywords") #string
    taglist.append("XMP:subject")
    keywordlist = []
    tagged = False
    if istagged:
        taglist.append('XMP:tagged')
    try:
            for d in et.get_tags(files=filetoproc, tags=taglist):
                for k, v in d.items():
                    #logger.info(f"Dict: {k} = {v}")
                    if k != 'SourceFile' and k != 'XMP:Tagged':
                            #logger.info(k)
                            if isinstance(v, list):
                                for line in v:
                                        keywordlist.append(str(line).strip())
                                res[k] = keywordlist
                            else:
                                if ',' in v or ';' in v:
                                    # If either a comma or semicolon is present in the value
                                    res[k] = [tag.strip() for tag in re.split('[,;]', v)]  # Split the string into a list using commas and semicolons as delimiters, and remove leading/trailing spaces
                                    #res[k] = ';'.join(tags)  # Join the list elements using semicolon as the separator and assign to the key k in the dictionary res
                                else:
                                    res[k] = v
                            #logger.info("test")
                    elif k == 'XMP:Tagged':
                        #logger.info("test")
                        if v: tagged = True

            for key, value in res.items():
                if isinstance(value, list):
                    res[key] = list(set(value))
            if len(res) == 0: res = False
    except Exception as e:
        logger.error(f"Error in get_description_keywords_tag.  {e}")
        res = False
        tagged = True
    
    if istagged:
        return res,tagged
    else:
        return res

@timing_decorator
def fix_person_tag(inputvar):
    try:
        if 'person' in str(inputvar.lower()) and 'people' not in str(inputvar.lower()):
            logger.info(f" {inputvar} contains a person record")
            #if 'people' in str(result2).lower():
            #    logger.info("people and person in tag")
            #if 'people' in inputvar.lower():
            #inputvar = search_replace_case_insensitive('person','People',inputvar)

            matches = re.finditer('person', inputvar, flags=re.IGNORECASE)

            # Iterate through matches and replace in a case-sensitive manner
            for match in matches:
                inputvar = inputvar[:match.start()] + 'People' + inputvar[match.end():]

        #inputvar = inputvar.replace('/','\\')
        #inputvar = inputvar.replace('|','\\')
        #inputvar = inputvar.replace("'",'')
        #inputvar = inputvar.replace("{",'')
        #inputvar = inputvar.replace("}",'')
        #inputvar = inputvar.replace('\\\\','\\')
        #remove spaces on either side of a forward slash
        #inputvar = re.sub(r'\s*/\s*', '/', inputvar)
        #inputvar = re.sub(r'\s*\/\s*', '\\', inputvar)

        #logger.info(f"result is {inputvar}")

        return inputvar
    except Exception as e:
        logger.error(f"fix_person_tag {e}")

@timing_decorator
def tidy_tags(lst):
    #return [item.replace("'", "").replace('"', "").replace("{", "").replace("}", "") for item in lst]

    cleaned_lst = [item.replace("'", "").replace('"', "").replace("{", "").replace("}", "") for item in lst]
    if cleaned_lst != lst:
        return cleaned_lst
    else:
        return False
    
@timing_decorator
def filter_person_from_list(lst):
    personname = None
    peoplename = None
    personindex = None
    peopleindex = None

    for i, item in enumerate(lst):

        lst[i] = re.sub(r'\s*/\s*', '/', lst[i])

        if '\\' in lst[i]:
                lst[i] = (lst[i]).replace('\\','/')

        if '//' in lst[i]:
                 lst[i] = (lst[i]).replace('//','/')

        if '|' in lst[i]:
                 lst[i] = (lst[i]).replace('|','/')

        if 'Person' in lst[i]:
            lst[i] = lst[i].replace('Person','People')
            
        if lst[i].lower() == 'person':
            lst.pop(i)

        if lst[i].lower() == 'people':
            lst.pop(i)

    return lst  # Return the original list if no changes are made

def set_union(set1, set2):
    #Union: Returns a new set containing all the distinct elements present in both sets.
    return set1.union(set2)

def set_difference(set1, set2):
    #Difference: Returns a new set containing elements that are present in the first set but not in the second set.
    return set1.difference(set2)

def is_subset(set1, set2):
    #Subset : if a set is a subset of another set.
    return set2.issubset(set1)

def is_superset(set1, set2):
    #Superset: if a set is a superset of another set.
    return set1.issuperset(set2)

def is_disjoint(set1, set2):
    #Disjoint: Checks if two sets have no common elements.
    return set1.isdisjoint(set2)

def symmetric_difference_update(set1, set2):
    #Updates the set with the symmetric difference of itself and another set.
    set1.symmetric_difference_update(set2)
    return set1


@timing_decorator
def has_duplicates(lst):
    try:
        seen = set()
        cnt = 0
        dupes =[]
        for item in lst:
            if item in seen:
                #logger.info(f"DUPE !.  {item} seen before in {lst}")
                dupes.append(item)
                cnt += 1
                #return True
            seen.add(item)
        if cnt >0:
            logger.info(f"{cnt} duplicates in list. {dupes}")
            return True
        return False
    except Exception as e:
        logger.error(f"has_duplicates error : {e}")

@timing_decorator
def apply_description_keywords_tag(filetoproc,valuetoinsert=None,markasprocessed=False,dedupe=False):
    res = {}
    taglist =[  "IPTC:Keywords",#list
                "XMP:TagsList",#list
                "XMP:HierarchicalSubject",#list
                "XMP:Categories",#string
                "XMP:CatalogSets",#list
                "XMP:LastKeywordIPTC",#list
                "XMP:LastKeywordXMP",#list
                #"XMP-microsoft:LastKeywordIPTC",
                #"XMP-microsoft:LastKeywordXMP",
                #"MicrosoftPhoto:LastKeywordIPTC",
                "EXIF:XPKeywords", #string
                "XMP:Subject"]#string

    stringlist =[  
                "XMP:Categories",#string
                "EXIF:XPKeywords", #string
                "XMP:Subject"
                ]
    seperatorstr = ";"
    mintaglength=3
    tagged = False
    forcetag = False
    forcewrite = False
    parentfolder = None

    if valuetoinsert != None:
        if isinstance(valuetoinsert, list):
            valuetoinsert = [str(value).replace("'", "") for value in valuetoinsert]
            keywordlist = valuetoinsert
        else:
            if ',' in valuetoinsert:
                keywordlist = valuetoinsert.split(',')
            elif ';' in valuetoinsert:
                keywordlist = valuetoinsert.split(';')

    #add any custom tags
    if add_parent_folder_as_tag == True:
        parentfolder = os.path.basename(os.path.dirname(filetoproc))
    if custom_tag != None:
        if isinstance(custom_tag, list):
            keywordlist.extend(custom_tag)
        else:
            keywordlist.append(custom_tag)
    if parentfolder != None:
        if add_parent_folder_as_people_tag == True:
            keywordlist.append('People/' + parentfolder)
        elif add_parent_folder_as_people_tag == False:
            keywordlist.append(parentfolder)
    if not markasprocessed:
        taglist.append('XMP:tagged')
    #end custom tags

    split_string_set = set(sorted(keywordlist))
    process = {}
    for each in taglist:
        process[each] = 0
    
    try:
        test2 =  et.get_metadata(files=filetoproc)
        exiftaglist =  et.get_tags(files=filetoproc, tags=taglist)
        logger.info(f"{et.last_stdout}")
        test = exiftaglist[0]
        #logger.info(f"get_tags output: {et.last_stdout}")
    except Exception as e:
        logger.error(f"Error apply_description_keywords_tag get metadata {e}")

 #   if '232.jpg' in filetoproc.lower():
 #       print("test")
    
    if (len(test) < len(taglist) and markasprocessed) or (len(test) <(len(taglist)-1) and not markasprocessed): #should be 9 returned.  MY 8 and SourceFile
        logger.info("not enough tags defined in image")
        for each in taglist:
            if each not in test and each != 'XMP:tagged' and (each not in stringlist):
                logger.info(f"tag {each} does not exist in metadata for {filetoproc}")
                res[each] = list(set([value for value in keywordlist if value]))
                process[each] += len(res[each])
            if (each in stringlist) and len(taglist) >0 :
                #final = list(copyofkeywordlist)
                logger.info(f"Tags ({keywordlist}) need adding to {each}")
                res[each] = seperatorstr.join(keywordlist)
        print("added all tags to blank")
        forcewrite = True

    elif keywordlist != None:
        try:
            for d in exiftaglist:
                for k, v in d.items():
                    logger.info(f"{type(v)}: {k} = {v}")
                    allkeywordsincpotentialdupes = []    
                    copyofkeywordlist = []
                    tags2 = set()
                    if k != 'SourceFile' and k != 'XMP:Tagged':               
                        if isinstance(v, list) or isinstance(v, dict):
                            logger.info(f"{type(v)}.  {k}.  {v}")
                            for line in v:
                                if isinstance(line, (int, float)):
                                    forcewrite = True
                                    continue
                                line = str(line).replace("|","/")
                                if ',' in line and seperatorstr != ',':
                                    forcewrite = True
                                if ';' in line and seperatorstr != ';':
                                    forcewrite = True
                                if ',' in str(line) or ';' in str(line):
                                    logger.info("List with ; or '. csv {k} line {line} is {v}")
                                    tags2 = [tag1.strip() for tag1 in re.split('[,;]', str(line))]  # Split the string into a list using commas and semicolons as delimiters, and remove leading/trailing spaces
                                else:
                                    if len(line) >2:
                                        tags2.add(str(line))
                        else:
                            if isinstance(v, (int, float)):
                                forcewrite = True
                                continue
                            v = str(v).replace("|","/")
                            logger.info(f"Not a list:{type(v)} {k}.. {v}")

                            if ',' in v and seperatorstr != ',':
                                forcewrite = True
                                forcetag = True
                            if ';' in v and seperatorstr != ';':
                                forcewrite = True
                                forcetag = True

                            if ',' in v or ';' in v:
                                logger.info(f"NOT a List. {k}.  {v} needs splitting")
                                if 'Categories' in v:
                                    logger.debug("categories detected")
                                    v = v.replace('<Categories>','').replace('<Category Assigned=1>','').replace('<Category Assigned=0>','').replace('</Category>',';').replace('</Categories>','')
                                if k not in stringlist:
                                    logger.info(f"{k} is a string.  Should be a list.  forcing retag as list. {v}")
                                    forcetag = True
                                    forcewrite = True
                                tags2 = [tag2.strip() for tag2 in re.split('[,;]', v)]  # Split the string into a list using commas and semicolons as delimiters, and remove leading/trailing spaces
                            else:
                                #empty value. Populate it
                                logger.info("Not a list and no ; or , value.  probably a single value")
                                #if len(v) == 0:
                                if k in stringlist:
                                    #They're all lists apart from XPKeywords
                                    logger.info(f"4.List. {k}")
                                    #copyofkeywordlist.extend([v] for item in copyofkeywordlist if item != v)
                                    
                                    #This should be a list so force retag
                                    forcetag = True
                                    forcewrite = True
                                    
                                    tags2 = [tag2.strip() for tag2 in re.split('[,;]', v)]  # Split the string into a list using commas and semicolons as delimiters, and remove leading/trailing spaces

                                else:
                                    logger.info("this should be EXIF:XPKeywords or XMP:Categories")
                                    tags2.add(str(v))
                                    #seen = set()
                                    #if v not in keywordlist:
                                    #    logger.info(f"5.")
                                    #    copyofkeywordlist = [x for x in keywordlist if x not in seen and not seen.add(x)]


                                    #for val in valuetoinsert:
                                    #        if val not in v:

                        result = tidy_tags(list(tags2))
                        if result != False:
                            logger.info("List was tidied.")
                            forcetag = True
                            copyofkeywordlist = set(result)
                        else:
                            original_set = set(sorted(tags2))
                            #            copyofkeywordlist.append(val)
                            unique_elements = split_string_set.difference(original_set)
                            unique_elements = {x for x in unique_elements if x != ''}
                            if len(unique_elements) >0:
                                logger.info(f"Unique Elements: {unique_elements}")
                                copyofkeywordlist = set_union(unique_elements,split_string_set)
                            if forcetag == True:
                                copyofkeywordlist = set_union(unique_elements,split_string_set)
                                forcetag = False

                        copyofkeywordlist = sorted(list(copyofkeywordlist))

                        if RemovePersonIfPeoplePresent:
                            copyofkeywordlist = filter_person_from_list(copyofkeywordlist)

                        copyofkeywordlist = set([value for value in copyofkeywordlist if value])

                        try:
                            process[k] += len(copyofkeywordlist)
                            if len(copyofkeywordlist) > 0:
                                print(f"{filetoproc} needs {unique_elements} adding to {k}. current tags are: {original_set}")
                        except Exception as e:
                            logger.error(f"add to array exception: {e}")

                        #if len(copyofkeywordlist) >0 and ((',' in v or ';' in v) or k =="EXIF:XPKeywords") :
                        if ((k in stringlist) and len(copyofkeywordlist) >0) or forcewrite == True:
                            #final = list(copyofkeywordlist)
                            logger.info(f"STRING: Tags ({copyofkeywordlist}) need adding to {k}")
                            res[k] = seperatorstr.join(copyofkeywordlist)
                        elif len(copyofkeywordlist) >0 :
                            final = sorted(list(copyofkeywordlist))
                            logger.info(f"DICT: Tags ({final}) need adding to {k}")
                            res[k] = final
                        else:
                            logger.info(f"No tags need adding to {k}")

                    elif k == 'XMP:Tagged':
                        if v: tagged = True
        except Exception as e:
            logger.error(f"Error creating tag list {e}")

    if any(value != 0 for value in process.values()) or (markasprocessed and not tagged) or forcewrite == True:
    #Only modify those with updated tags
        logger.info(f"{filetoproc} needs updating {sum(value for value in process.values())} tags need updating {res}")
        if markasprocessed:
            res['XMP:tagged'] = "true"
        try:
            et.set_tags(filetoproc, tags=res,params=["-P", "-overwrite_original"])
            logger.info(f"{et.last_stdout}")
            if '1 image files updated' not in et.last_stdout:
                logger.error(f"Error !!! {filetoproc}. {res} {et.last_stdout}")
        except Exception as e:
            logger.error(f"Error set_tags {e}")
    elif (markasprocessed and tagged):
        logger.info(f"{filetoproc} marked as processed but already tagged as processed.  ")
    else:
        logger.info(f"No modifications required to {filetoproc}")
        
    #else:
    #    deletestring = ""
    #    for each in taglist:
    #        deletestring = deletestring + f"-{each}= "
        #et.execute(deletestring, filetoproc)

    return True


@timing_decorator
def blip_large(imagepath,model='small'):

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
    logger.info(processor.decode(out[0], skip_special_tokens=True))

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)

    res = processor.decode(out[0], skip_special_tokens=True)
    return res


@timing_decorator
def write_pnginfo(filename,tags):
    try:
        if os.path.exists(filename):
            writefile = False
            image = Image.open(filename)
            metadata = PngImagePlugin.PngInfo()
            inferencefound = False
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    if key == 'exif':
                        logger.info("exif data breaks the file.  Skip {filename}")
                        continue
                    elif key == 'parameters':
                        logger.info(f"Stable Diffusion file. {filename}: {value}")
                        metadata.add_text(key, value)
                        sd = True
                    elif key =='Inference':
                        logger.info(f"inference text already exists. {filename}: {value}")
                        inferencefound = True
                        metadata.add_text(key,value)
                    else:
                        logger.info(f"Other: {key}.  {value}")
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
                    logger.error(f"error write_pnginfo {e}")
                os.utime(filename, (original_atime, original_mtime))
                logger.info(f"atime and mtime restored.")
    except Exception as e:
        logger.error(f"write_pnginfo: Error {e}")

@timing_decorator
def search_replace_case_insensitive(search_pattern, replace_text, input_string):
    # Perform case-insensitive search
    matches = re.finditer(search_pattern, input_string, flags=re.IGNORECASE)

    # Iterate through matches and replace in a case-sensitive manner
    result = input_string
    for match in matches:
        result = result[:match.start()] + replace_text + result[match.end():]

    return result       

@timing_decorator
def modify_exif_comment(filename, tags, command, new_value=None, tagname= None):
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

        # Convert single tag to a list
        if isinstance(tags, str):
            tags = [tags]

        if exifdata == None:
            logger.info("No exifdata")
            found = False
        else:
            # Use a custom tag (you can modify this based on your requirements)
            found = False
            if tagname is not None:
                    for pil_tag, pil_tag_name in TAGS.items():
                        if pil_tag_name == tagname:
                            #custom_tag = hex(pil_tag_name)
                            custom_tag = pil_tag
                            logger.info(f"using {pil_tag} for {tagname} tag")
                            found = True
                            break
        if found == False or tagname == None:
            # 40094:0x9C9E:'XPKeywords'
            logger.info("No exifdata or tagname = None.  Using XPKeywords for tag")
            #custom_tag = 0x9C9E
            custom_tag = 40094

        # Check if the custom tag is present in the Exif data
        if custom_tag not in exifdata:
            # Custom tag doesn't exist, add it with an initial value
            
            exifdata[custom_tag] = ''.encode('utf-16le')
            #exifdata[custom_tag] = ''.encode('utf-16')
            logger.info("image doesn't currently have any tags")
            current_tags = []
        else:
            does_image_have_tags = True
            logger.info("image currently has tags")

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
                logger.info("current_tags is there, but has no tags in")

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
                logger.info("does_image_have_tags is False, skipping removal.")

        elif command == 'show':
            # Return the list of tags or None if empty
            logger.info(f"Exif tags {command}ed successfully.")
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
                        logger.info("does_image_have_tags is False, skipping update.")                            
                else:
                    logger.info("Missing new_value for 'update' command.")
                    return
        elif command == 'clear':
            # Clear all tags
            tags_list = []
        elif command == 'count':
            # Get the count of tags
            logger.info(f"Exif tags {command} completed successfully.")
            if does_image_have_tags == True:
                return len(tags_list)
            else:
                return 0
        elif command == 'search':
            # Check if a specific tag exists
            if does_image_have_tags == True:
                logger.info(f"Exif tags {command}ed successfully.")
                return any(tag in current_tags for tag in tags)
            else:
                return ''
        else:
            logger.info("Invalid command. Please use 'add', 'remove', 'show', 'update', 'clear', 'count', or 'search'.")
            return

        # Check if the tags have changed
        if does_image_have_tags == True:
            #remove dupes
            new_tags_set = set(tags_list)
            #remove empty/null
            new_tags_set = {value for value in new_tags_set if value}

        if does_image_have_tags == False or len(tags_list) > 0:
            if does_image_have_tags == False:
                logger.info(f"no tags originally.  Need to add tags {str(list(tags_list))}.")
            else:
                logger.info(f"need to add tags {str(list(tags_list))}.  Current tags are {str(list(current_tags))}")

        #if updated_tags_string != tags_string_concat:
            # Encode the modified tags string and update the Exif data
            # Join the modified tags list into a string
            updated_tags_string = ';'.join(tags_list)

            #exifdata[custom_tag] = updated_tags_string.encode('utf-16')
            exifdata[custom_tag] = updated_tags_string.encode('utf-16le')

            # Save the image with updated Exif data
            image.save(filename, exif=exifdata)
            logger.info(f"Exif tags {command}ed successfully to {filename}.")
            os.utime(filename, (original_atime, original_mtime))
            logger.info(f"atime and mtime restored.")
        else:
            logger.info(f"No changes in tags for file {filename}. File not updated.")
    else:
        logger.info(f"File not found: {filename}")

@timing_decorator
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

        # Convert single tag to a list
        if isinstance(tags, str):
            tags = [tags]

        if exifdata == None:
            logger.info("No exifdata")
            found = False
        else:
            # Use a custom tag (you can modify this based on your requirements)
            found = False
            if tagname is not None:
                    for pil_tag, pil_tag_name in TAGS.items():
                        if pil_tag_name == tagname:
                            #custom_tag = hex(pil_tag_name)
                            custom_tag = pil_tag
                            logger.info(f"using {pil_tag} for {tagname} tag")
                            found = True
                            break
        if found == False or tagname == None:
            # 40094:0x9C9E:'XPKeywords'
            logger.info("No exifdata or tagname = None.  Using XPKeywords for tag")
            #custom_tag = 0x9C9E
            custom_tag = 40094

        # Check if the custom tag is present in the Exif data
        if custom_tag not in exifdata:
            # Custom tag doesn't exist, add it with an initial value
            
            exifdata[custom_tag] = ''.encode('utf-16le')
            #exifdata[custom_tag] = ''.encode('utf-16')
            logger.info("image doesn't currently have any tags")
            current_tags = []
        else:
            does_image_have_tags = True
            logger.info("image currently has tags")

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
                logger.info("current_tags is there, but has no tags in")

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
                logger.info("does_image_have_tags is False, skipping removal.")

        elif command == 'show':
            # Return the list of tags or None if empty
            logger.info(f"Exif tags {command}ed successfully.")
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
                        logger.info("does_image_have_tags is False, skipping update.")                            
                else:
                    logger.info("Missing new_value for 'update' command.")
                    return
        elif command == 'clear':
            # Clear all tags
            tags_list = []
        elif command == 'count':
            # Get the count of tags
            logger.info(f"Exif tags {command} completed successfully.")
            if does_image_have_tags == True:
                return len(tags_list)
            else:
                return 0
        elif command == 'search':
            # Check if a specific tag exists
            if does_image_have_tags == True:
                logger.info(f"Exif tags {command}ed successfully.")
                return any(tag in current_tags for tag in tags)
            else:
                return ''
        else:
            logger.info("Invalid command. Please use 'add', 'remove', 'show', 'update', 'clear', 'count', or 'search'.")
            return

        # Check if the tags have changed
        if does_image_have_tags == True:
            #remove dupes
            new_tags_set = set(tags_list)
            #remove empty/null
            new_tags_set = {value for value in new_tags_set if value}

        if does_image_have_tags == False or len(tags_list) > 0:
            if does_image_have_tags == False:
                logger.info(f"no tags originally.  Need to add tags {str(list(tags_list))}.")
            else:
                logger.info(f"need to add tags {str(list(tags_list))}.  Current tags are {str(list(current_tags))}")

        #if updated_tags_string != tags_string_concat:
            # Encode the modified tags string and update the Exif data
            # Join the modified tags list into a string
            updated_tags_string = ';'.join(tags_list)

            #exifdata[custom_tag] = updated_tags_string.encode('utf-16')
            exifdata[custom_tag] = updated_tags_string.encode('utf-16le')

            # Save the image with updated Exif data
            image.save(filename, exif=exifdata)
            logger.info(f"Exif tags {command}ed successfully to {filename}.")
            os.utime(filename, (original_atime, original_mtime))
            logger.info(f"atime and mtime restored.")
        else:
            logger.info(f"No changes in tags for file {filename}. File not updated.")
    else:
        logger.info(f"File not found: {filename}")

@timing_decorator
def load_image_in_thread(image_path):

    try:
        image1 = Image.open(image_path).resize((400, 300), Image.LANCZOS)
    except Exception as e:
        logger.error(f'{e}')
        prepend_string_to_filename(image_path,'corrupt_')
        return None

    return ImageTk.PhotoImage(image1)

@timing_decorator
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

@timing_decorator
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
        logger.info(f"model path is {model_path}")

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

@timing_decorator
def image_to_wd14_tags(filename,modeltouse='wd14-vit-v2') \
        -> Tuple[Mapping[str, float], str, Mapping[str, float]]:
    
    try:
        image = Image.open(filename)
        logger.info("image: " + filename + " successfully opened.  Continue processing ")
    except Exception as e:
        logger.error("Processfile Exception1: " + " failed to open image : " + filename + ". FAILED Error: " + str(e) + ".  Skipping")
        return None

    try:
        logger.info(modeltouse)
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
        logger.error(f"Exception getting tags from image {filename}.  Error: {e}" )
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
            logger.info("hi")
            logger.info(image_path)
            logger.info(str(test))
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

            logger.info(f'{image_path} {text_set}')
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
                logger.info("image was corrupt")
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

CHECK_ISTAGGED= True
tag_as_processed = False
gpu = False
gui = True
interrogateImage = True
CheckForPersonsNameInTags = False
RemovePersonIfPeoplePresent = True
tidyuptags = True
timing_debug = True
add_parent_folder_as_tag = False
add_parent_folder_as_people_tag = False
custom_tag = None
defaultdir = '/folder/to/process'

current_os = get_operating_system()

localoverridesfile = os.path.join(get_script_path(), "localoverridesfile_" + get_script_name() + '_' + current_os + '.py')
log_file_path =  os.path.join(get_script_path(),get_script_name() + '.log')
errorlog_file_path =  os.path.join(get_script_path(),get_script_name() + '_error.log')

logger = utils.setup_logging(log_file_path, errorlog_file_path, log_level='info')

if current_os == "Windows":
    logger.info("Running on Windows")

    import psutil
    # Set process priority to below normal
    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

elif current_os == "Linux":
    logger.info("Running on Linux")
    current_niceness = os.nice(0)
    print("Current niceness value:", current_niceness)
    #os.nice(-10)

if os.path.exists(localoverridesfile):
    exec(open(localoverridesfile).read())
    #apikey = apikey
    #logger.info("API Key:", apikey)
    logger.info("local override file is " + localoverridesfile)

else:
    logger.info("local override file would be " + localoverridesfile)

RE_SPECIAL = re.compile(r'([\\()])')

if CheckForPersonsNameInTags:
    import nltk
    from nltk.tag.stanford import StanfordNERTagger 
    nltk.download('punkt')

    #sudo apt-get install default-jre-headless
    #wget https://nlp.stanford.edu/software/stanford-ner-4.2.0.zip
    #unzip stanford-ner-4.2.0.zip
    #mkdir stanford-ner
    #cp stanford-ner-4.2.0/stanford-ner.jar stanford-ner/stanford-ner.jar
    #cp stanford-ner-4.2.0/classifiers/english.all.3class.distsim.crf.ser.gz stanford-ner/english.all.3class.distsim.crf.ser.gz
    #cp stanford-ner-4.2.0/classifiers/english.all.3class.distsim.prop stanford-ner/english.all.3class.distsim.prop
    #rm -rf stanford-ner-4.2.0 stanford-ner-4.2.0.zip

    ##wget http://nlp.stanford.edu/software/stanford-ner-2014-08-27.zip
    ##unzip stanford-ner-2014-08-27.zip
    ##mkdir stanford-ner
    ##cp stanford-ner-2014-08-27/stanford-ner.jar stanford-ner/stanford-ner.jar
    ##cp stanford-ner-2014-08-27/classifiers/english.all.3class.distsim.crf.ser.gz stanford-ner/english.all.3class.distsim.crf.ser.gz
    ##cp stanford-ner-2014-08-27/classifiers/english.all.3class.distsim.prop stanford-ner/english.all.3class.distsim.prop
    ##rm -rf stanford-ner-2014-08-27 stanford-ner-2014-08-27.zip
    st = StanfordNERTagger (get_script_path() + '/stanford-ner/english.all.3class.distsim.crf.ser.gz', get_script_path() + '/stanford-ner/stanford-ner.jar')

ci = None
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024,expandable_segments:True"

et = ExifToolHelper(logger=logger,common_args=['-G', '-n','-a','-P',"-overwrite_original",'-m'])
# Start ExifTool process

model_loaded = False
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
                'ViT-L-14': 'ViT-L-14/openai',
                'ViT-L-14': 'immich-app/ViT-L-14__openai',
                'ViT-H-14': 'ViT-H-14/laion2b_s32b_b79',
                'ViT-H-14': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
                'wd14': 'wd14-convnext',
                'wd14': 'saltacc/wd-1-4-anime',
                'wd' : 'SmilingWolf/wd-v1-4-vit-tagger-v2',
                'blip-base': 'Salesforce/blip-image-captioning-base',   # 990MB
                'blip-large': 'Salesforce/blip-image-captioning-large', # 1.9GB
                'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',              # 15.5GB
                'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',      # 15.77GB
                'git-large-coco': 'microsoft/git-large-coco'           # 1.58GB
                }

    for root, dirs, files in os.walk(defaultdir):
        print("Processing directory:", root)
        dirs.sort()
        files.sort()
        for filename in files:
            fullpath = os.path.join(root,filename)
            logger.info(f"{fullpath} - Processing")
            if filename.lower().endswith(('.jpg', '.jpeg','.png')):
                if CHECK_ISTAGGED:
                    tmp,istagged = get_description_keywords_tag(fullpath,True)
                else:
                    tmp = get_description_keywords_tag(fullpath)
                    istagged = False
                
                if not istagged:
                    logger.info(f"{fullpath} - Not tagged continuing")
                    result = None
                
                    # for each,desc in modelarray.items():
                    #     logger.info("using: " + each)
                    #     processor = BlipProcessor.from_pretrained(desc)
                    #     model = BlipForConditionalGeneration.from_pretrained(desc)
                    #     image = Image.open(fullpath).convert('RGB')
                    #     inputs = processor(image, return_tensors="pt")
                    #     out = model.generate(**inputs)
                    #     logger.info(f"{fullpath}. {each} {processor.decode(out[0], skip_special_tokens=True)}")
                    #     logger.info("press a key to continue")
                    #     input()
                    # break

                    #result = ddb(fullpath)
                    #result = image_to_wd14_tags(fullpath,'wd14-vit-v2')
                    #logger.info(f"{fullpath} . {str(result)} . wd14-vit-v2") 
                    #result = image_to_wd14_tags(fullpath,'wd14-convnext')
                    #logger.info(f"{fullpath} . {str(result)} . wd14-convnext")#377MB model.onnx
                    #result = image_to_wd14_tags(fullpath,'wd-v1-4-moat-tagger-v2')
                    #logger.info(f"{fullpath} . {str(result)} . wd-v1-4-moat-tagger-v2")#377MB model.onnx
                    #result = image_to_wd14_tags(fullpath,'wd-v1-4-swinv2-tagger-v2')
                    #logger.info(f"{fullpath} . {str(result)} . wd-v1-4-swinv2-tagger-v2")#377MB model.onnx
                    #result = image_to_wd14_tags(fullpath,'wd-v1-4-convnext-tagger-v2')
                    #logger.info(f"{fullpath} . {str(result)} . wd-v1-4-convnext-tagger-v2")#377MB model.onnx
                    #logger.info(f"{fullpath} . {str(result)} . wd-v1-4-convnextv2-tagger-v2")#377MB model.onnx
                    #result = image_to_wd14_tags(fullpath,'wd-v1-4-vit-tagger-v2')
                    #logger.info(f"{fullpath} . {str(result)} . wd-v1-4-vit-tagger-v2")#377MB model.onnx

                    if interrogateImage == True:
                        try:
                            if gpu == True:
                                result = use_GPU_interrogation(fullpath)
                            else:
                                result = image_to_wd14_tags(fullpath,'wd-v1-4-convnextv2-tagger-v2')

                            if result == None:
                                logger.info("nothing returned from interrogation")
                            else:
                                print(f"  interrogator output was {result[1]}")
                        except Exception as e:
                            logger.error(f"interrogation failed.  {e}")
                         
                    #result = image_to_wd14_tags(fullpath,'ViT-L-14/openai')
                    #logger.info(f"{fullpath} . {str(result)} . ViT-L-14/openai")
                    #result = image_to_wd14_tags(fullpath,'ViT-H-14/laion2b_s32b_b79')
                    #logger.info(f"{fullpath} . {str(result)} . ViT-H-14/laion2b_s32b_b79")
                    #result = image_to_wd14_tags(fullpath,'ViT-L-14/openai')
                    #logger.info(f"{fullpath} . {str(result)} . ViT-L-14/openai")

                    #test = blip2_opt_2_7b(fullpath)
                    #test = blip_large(fullpath)
                    #test = unumcloud(fullpath)
                    #test = nlpconnect(fullpath)
                    #logger.info(f"{fullpath} . {str(test)}")
                    #exit()
                    
                    if result is not None:
                        keywords = result[1]   

                        if CheckForPersonsNameInTags == True:
                            test = ' '.join(keywords)
                            isperson = False
                            if is_person(test):
                                isperson = True
                                keywords.append("ISPERSON")

                        if len(keywords) > 0:
                            #logger.info(str(result2))
                            #tagname = 'XPKeywords'
                            #tagname = 'EXIF:XPKeywords'
                            try:
                                if filename.lower().endswith(('.png')):
                                    write_pnginfo(fullpath, keywords)
                                    apply_description_keywords_tag(fullpath,keywords,tag_as_processed,True)
                                elif filename.lower().endswith(('.jpg', '.jpeg')):
                                    #modify_exif_tags(fullpath, result2, 'add',None,tagname)
                                    apply_description_keywords_tag(fullpath,keywords,tag_as_processed,True)

                            except Exception as e:
                                logger.error(f"writing tags failed.  {e}")
                        else:
                            logger.info("stuff detected but not relevant/length 0")
                    else:
                        logger.info(f"nothing detected for {fullpath}.  Odd")
                else:
                    logger.info(f"{fullpath} - Tagged as processed.  Skipping")
                logger.info(f"{fullpath} - Processing complete")
            else:
                logger.info(f"{fullpath} - Unsupported filetype.  Not an image.")

et.terminate()
