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
import torch

import pandas as pd
import cv2
import numpy as np
from typing import Mapping, Tuple, Dict
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession

from exiftool import ExifToolHelper

#importing libraries
import os
import re

#import glob

from PIL.ExifTags import TAGS
from PIL import Image, ImageTk
#below for pngs
from PIL import PngImagePlugin, Image
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


def is_person(snippet):

    try:
        for sent in nltk.sent_tokenize(snippet):
            tokens = nltk.tokenize.word_tokenize(sent)
            tags = st.tag(tokens)
            for tag in tags:
                if tag[1]=='PERSON': 
                    print(f"{tag[0]} is a persons name !!!!!!")
                    return True
                #elif tag[1]=='ORGANIZATION':
                #    print(f"{tag} is probably a persons name !!!!!!{tag[1]}")
                #    return True  
        return
    except Exception as e:
        print(f"Error in isperson{e}")

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            import gc
            #print("mem before")
            #print(torch.cuda.memory_summary(device=None, abbreviated=False))
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            #del variables

            #print("mem after")
            #print(torch.cuda.memory_summary(device=None, abbreviated=False))

def low_vram():
    if torch.cuda.is_available():
        vram_total_mb = torch.cuda.get_device_properties('cuda').total_memory / (1024**2)
        vram_info = f"GPU VRAM: **{vram_total_mb:.2f}MB**"
        if vram_total_mb< 8:
            vram_info += "<br>Using low VRAM configuration"
            print(f"{vram_info}")
    if vram_total_mb <= '4': return False 
    return True

def return_vram():
    if torch.cuda.is_available():
        vram_total_mb = torch.cuda.get_device_properties('cuda').total_memory / (1024**2)
    return vram_total_mb


def load(clip_model_name):
    global ci
    if ci is None:
        print(f"Loading CLIP Interrogator {clip_interrogator.__version__}...")

        config = Config(
            cache_path = 'models/clip-interrogator',
            clip_model_name=clip_model_name,
        )

        if low_vram:
            print("low vram")
            config.apply_low_vram_defaults()
            config.chunk_size = 512
        ci = Interrogator(config)

    if clip_model_name != ci.config.clip_model_name:


        ci.config.clip_model_name = clip_model_name
        torch_gc()
        #with Timer() as modelloadtime:
        ci.load_clip_model()
        #print(f"loading model took {modelloadtime.last} to load")
        #return res, modelloadtime.last


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

def prepend_string_to_filename(fullpath, prefix):
    # Split the full path into directory and filename
    directory, filename = os.path.split(fullpath)

    # Prepend the prefix to the filename
    new_filename = f"{prefix}{filename}"

    # Join the directory and the new filename to get the updated full path
    new_fullpath = os.path.join(directory, new_filename)

    return new_fullpath

def nlpconnect(fileinput):
    #nlpconnect
    from transformers import pipeline

    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    return image_to_text(fileinput)

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
    print(processor.decode(out[0], skip_special_tokens=True).strip())

def use_GPU_interrogation(image_path,model_name="ViT-L-14/openai"):
    
    load("ViT-L-14/openai")
        #models = list_clip_models()
    #print(f"supported models are {models}")
    print("load image")
    image = Image.open(image_path).convert('RGB')
    print("convert RGB")
    #ci = Interrogator(Config(clip_model_name=model_name))
    #print("create CI")
    res = ci.interrogate(image)
    print ("interrogation complete")
    print(res)
    return (res)


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
        with ExifToolHelper() as et:
                for d in et.get_tags(files=filetoproc, tags=taglist):
                    for k, v in d.items():
                        #print(f"Dict: {k} = {v}")
                        if k != 'SourceFile' and k != 'XMP:Tagged':
                                #print(k)
                                if isinstance(v, list):
                                    for line in v:
                                            keywordlist.append(str(line).strip())
                                    res[k] = keywordlist
                                else:
                                    if ',' in v or ';' in v:
                                        # If either a comma or semicolon is present in the value
                                        tags = [tag.strip() for tag in re.split('[,;]', v)]  # Split the string into a list using commas and semicolons as delimiters, and remove leading/trailing spaces
                                        res[k] = ';'.join(tags)  # Join the list elements using semicolon as the separator and assign to the key k in the dictionary res
                                    else:
                                        res[k] = v
                                #print("test")
                        elif k == 'XMP:Tagged':
                            #print("test")
                            if v: tagged = True

                for key, value in res.items():
                    if isinstance(value, list):
                        res[key] = list(set(value))
                if len(res) == 0: res = False
    except Exception as e:
        print(f"Error {e}")
        res = False
        tagged = True
    
    if istagged:
        return res,tagged
    else:
        return res

def fix_person_tag_dict(inputvar):

    islist = False
    if 'person' in str(inputvar).lower() and 'ISPERSON' not in str(inputvar):
        print("contains a person record")
        #if 'people' in str(result2).lower():
        #    print("people and person in tag")
        updated_result = []

        if isinstance(inputvar, list):
            islist = True
        else:
            if ';' in inputvar:
                delim = ';'
            elif ',' in inputvar:
                delim = ','
            else:
                print("unknown delimiter")
                return None
            inputvar = [tag.strip() for tag in re.split('[,;]', str(inputvar))]

        for each in inputvar:
            if 'people' in each.lower():
                each = search_replace_case_sensitive('person','People',each)
            each = each.replace('\\','/')
            #remove spaces on either side of a forward slash
            each = re.sub(r'\s*/\s*', '/', each)
            updated_result.append(each)
        inputvar = updated_result
        inputvar = list(set(inputvar))

        if islist == True:
            return inputvar
        else:
            return delim.join(inputvar)
    else:
        return inputvar
  
def fix_person_tag(inputvar):
    try:
        if 'person' in inputvar.lower() and 'ISPERSON' not in inputvar:
            print(f" {inputvar} contains a person record")
            #if 'people' in str(result2).lower():
            #    print("people and person in tag")
            #if 'people' in inputvar.lower():
            inputvar = search_replace_case_sensitive('person','People',inputvar)

        inputvar = inputvar.replace('\\','/')
        inputvar = inputvar.replace('|','/')
        inputvar = inputvar.replace("'",'')
        inputvar = inputvar.replace("{",'')
        inputvar = inputvar.replace("}",'')
        #remove spaces on either side of a forward slash
        inputvar = re.sub(r'\s*/\s*', '/', inputvar)
        #print(f"result is {inputvar}")

        return inputvar
    except Exception as e:
        print(f"fix_person_tag {e}")


def has_duplicates(lst):
    try:
        seen = set()
        cnt = 0
        dupes =[]
        for item in lst:
            if item in seen:
                #print(f"DUPE !.  {item} seen before in {lst}")
                dupes.append(item)
                cnt += 1
                #return True
            seen.add(item)
        if cnt >0:
            print(f"{cnt} duplicates in list. {dupes}")
            return True
        return False
    except Exception as e:
        print(f"has_duplicates {e}")
def apply_description_keywords_tag(filetoproc,valuetoinsert=None,markasprocessed=False,dedupe=False):
    res = {}
    taglist =[]
    taglist.append("IPTC:Keywords")#list
    taglist.append("XMP:TagsList")
    taglist.append("XMP:HierarchicalSubject")
    taglist.append("XMP:Categories")
    taglist.append("XMP:CatalogSets")
    taglist.append("XMP:LastKeywordXMP")
    taglist.append("EXIF:XPKeywords") #string
    taglist.append("XMP:Subject")

    mintaglength=3
    tagged = False
    if not markasprocessed:
        taglist.append('XMP:tagged')

    if valuetoinsert != None:
        if isinstance(valuetoinsert, list):
            valuetoinsert = [value.replace("'", "") for value in valuetoinsert]
            keywordlist = valuetoinsert
        else:
            if ',' in valuetoinsert:
                keywordlist = valuetoinsert.split(',')
            elif ';' in valuetoinsert:
                keywordlist = valuetoinsert.split(';')
    elif valuetoinsert == 'del':
        keywordlist = None

    process = {}
    for each in taglist:
        process[each] = 0
    
    dedupedkeywords = {}
    dupes = False

    try:
        with ExifToolHelper() as et:
            #exiftaglist =  et.get_tags(files=filetoproc,tags=None)
            exiftaglist =  et.get_tags(files=filetoproc, tags=taglist)
            test = exiftaglist[0]
            #print(f"get_tags output: {et.last_stdout}")
    except Exception as e:
        print(f"Error a {e}")
    
    if len(test) <9: #should be 9 returned.  MY 8 and SourceFile
        print("not enough tags defined in image")
        for each in taglist:
            if each not in test and each != 'XMP:tagged':
                print(f"tag {each} does not exist in metadata for {filetoproc}")
                if each != 'EXIF:XPKeywords':
                    res[each] = keywordlist  
                elif each == 'EXIF:XPKeywords':
                    res[each] = ';'.join(keywordlist)
                process[each] += len(keywordlist)
                
    if keywordlist != None:
        for d in exiftaglist:
            #if '18983' in str(d):
            #print(f"{str(d)}")
            #    print("debug")
            for k, v in d.items():
                #print(f"Dict: {k} = {v}")
                allkeywordsincpotentialdupes = []    
                copyofkeywordlist = []
                if k != 'SourceFile' and k != 'XMP:Tagged':
                    if isinstance(v, list):
                        #print(f"list {k}.  {v}")
                        for line in v:
                            if ',' in str(line) or ';' in str(line):
                                #print("csv {k} line {line} is {v}")
                                tags = [tag.strip() for tag in re.split('[,;]', str(line))]  # Split the string into a list using commas and semicolons as delimiters, and remove leading/trailing spaces
                                for one in tags:
                                    valtoinsert = one.strip()
                                    if len(valtoinsert)>mintaglength:
                                        allkeywordsincpotentialdupes.append(valtoinsert)
                                        if valtoinsert not in v:
                                            valtoinsert = fix_person_tag(valtoinsert)
                                            copyofkeywordlist.append(valtoinsert)
                                            print(f"{valtoinsert} needs adding")
                                    else:
                                        print(f"remove {valtoinsert} as it's too short ")
                            elif line not in v:
                                #print(f"{k}.  {line} line not in {v}")
                                valtoinsert = str(line).strip()
                                if len(valtoinsert)>mintaglength:
                                    valtoinsert = fix_person_tag(valtoinsert)
                                    allkeywordsincpotentialdupes.append(valtoinsert)
                                    copyofkeywordlist.append(valtoinsert)
                            elif line in v:
                                #print(f"{k}. {line} linein {v}")
                                line = fix_person_tag(line)
                                allkeywordsincpotentialdupes.append(line)
                                #pass
                                #print(f"{line} already in {v}")
                            else:
                                print("shouldn't get here")
                        #print("p")
                        if len(copyofkeywordlist) >0:
                            #print("dedupe")
                            setb = set(copyofkeywordlist)
                            copyofkeywordlist += [str(item).strip() for item in keywordlist if len(str(item).strip()) > mintaglength and str(item).strip() not in setb]
                            print("Tags ({copyofkeywordlist}) need adding to {k}")
                            res[k] = copyofkeywordlist

                        #print("q")
                        if has_duplicates(allkeywordsincpotentialdupes):
                            print("has dupes")
                            dedupedkeywords[k] = set(allkeywordsincpotentialdupes)
                            dupes = True
                    else:
                        #print(f"Not a list {k}.. {v}")
                        if ',' in v or ';' in v:
                            tags = [tag.strip() for tag in re.split('[,;]', v)]  # Split the string into a list using commas and semicolons as delimiters, and remove leading/trailing spaces
                            #TODO  not even using tags here !
                            for val in valuetoinsert:
                                    if val not in v and len(val) >mintaglength:
                                        val = fix_person_tag(val)
                                        copyofkeywordlist.append(val)
                            if len(copyofkeywordlist) >0:
                                if has_duplicates(copyofkeywordlist):
                                    copyofkeywordlist = set(copyofkeywordlist)
                                res[k] = ';'.join(copyofkeywordlist)
                        else:
                            #empty value. Populate it
                            #print("empty")
                            if len(v) == 0:
                                if k != 'EXIF:XPKeywords':
                                    #They're all lists apart from XPKeywords
                                    for each in keywordlist:
                                        if len(each) >mintaglength:
                                            each = fix_person_tag(each)
                                            copyofkeywordlist.append(each)
                                            allkeywordsincpotentialdupes.append(each) 
                                    if has_duplicates(copyofkeywordlist):
                                        copyofkeywordlist = set(copyofkeywordlist)
                                    res[k] = copyofkeywordlist

                                else:
                                    print("this should be EXIF:XPKeywords")
                                    res[k] = ';'.join(keywordlist)
                            else:
                                #print("tt")
                                if k != 'EXIF:XPKeywords':
                                    for each in keywordlist:
                                        #print("each")
                                        if len(each) >mintaglength:
                                            each = fix_person_tag(each)
                                            copyofkeywordlist.append(each)
                                    if has_duplicates(copyofkeywordlist):
                                        copyofkeywordlist = set(copyofkeywordlist)
                                    res[k] = copyofkeywordlist
                                else:
                                    #print("datatoupdate")
                                    datatoupdate = []
                                    for each in keywordlist:
                                        datatoupdate.append(fix_person_tag(each))
                                    keywordlist = datatoupdate
                                    if has_duplicates(keywordlist):
                                        keywordlist = set(keywordlist)
                                    res[k] = ';'.join(keywordlist)
                        #print("f")
                        if has_duplicates(allkeywordsincpotentialdupes):
                            dedupedkeywords[k] = set(allkeywordsincpotentialdupes)
                            dupes = True
                    #print("rrr")
                    #print(f"{k} length of copyofkeywordlist is {len(copyofkeywordlist)}.  type is {type(copyofkeywordlist)}")
                    try:
                        process[k] += len(copyofkeywordlist)
                    except Exception as e:
                        print(f"exception: {e}")
                    #print("sss")
                elif k == 'XMP:Tagged':
                    #print("test")
                    if v: tagged = True

        #print("oo")

        if dedupe == True and dupes == True:
            #deduped = set(allkeywordsincpotentialdupes)
            #print("Dedupe checking")
            try:
                with ExifToolHelper() as et:
                    et.set_tags(filetoproc, tags=dedupedkeywords,params=["-P", "-overwrite_original"])
                print(f"dedupe set tag error {et.last_stdout}")
            except Exception as e:
                print(f"Error a {e}")
        elif dedupe == True and dupes == False:
            print("Dupe checking enabled and No dupes found")

        if any(value != 0 for value in process.values()):
        #Only modify those with updated tags
            print(f"needs updating {sum(value for value in process.values())} tags need updating {res}")
            if markasprocessed:
                res['XMP:tagged'] = "true"
            try:
                with ExifToolHelper() as et:
                    et.set_tags(filetoproc, tags=res,params=["-P", "-overwrite_original"])
                print(f"{et.last_stdout}")
            except Exception as e:
                print(f"Error b {e}")
            return True
        elif markasprocessed and not tagged:
            print(f"Force marked {filetoproc} as tagged due to config")
            res['XMP:tagged'] = True
            try:
                with ExifToolHelper() as et:
                    et.set_tags(filetoproc, tags=res,params=["-P", "-overwrite_original"])
                print(f"{et.last_stdout}")
                return True
            except Exception as e:
                print(f"Error c {e}")
        else:
            print(f"No modifications required to {filetoproc}")
        
    else:
        deletestring = ""
        for each in taglist:
            deletestring = deletestring + f"-{each}= "
        #et.execute(deletestring, filetoproc)

    return True


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


def search_replace_case_sensitive(search_pattern, replace_text, input_string):
    # Perform case-insensitive search
    matches = re.finditer(search_pattern, input_string, flags=re.IGNORECASE)

    # Iterate through matches and replace in a case-sensitive manner
    result = input_string
    for match in matches:
        result = result[:match.start()] + replace_text + result[match.end():]

    return result       

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

CHECK_ISTAGGED= True
tag_as_processed = False
gpu = True
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
st = StanfordNERTagger (get_script_path() + '/stanford-ner/english.all.3class.distsim.crf.ser.gz', get_script_path() + '/stanford-ner/stanford-ner.jar')
ci = None
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024,expandable_segments:True"

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
        for filename in files:
            fullpath = os.path.join(root,filename)
            print(f"{fullpath} - Processing")
            if filename.lower().endswith(('.jpg', '.jpeg','.png')):
                if CHECK_ISTAGGED:
                    tmp,istagged = get_description_keywords_tag(fullpath,True)
                else:
                    tmp = get_description_keywords_tag(fullpath)
                    istagged = False
                
                if not istagged:
                    print(f"{fullpath} - Not tagged continuing")
                    result = None
                
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
                    #print(f"{fullpath} . {str(result)} . wd-v1-4-convnextv2-tagger-v2")#377MB model.onnx
                    #result = image_to_wd14_tags(fullpath,'wd-v1-4-vit-tagger-v2')
                    #print(f"{fullpath} . {str(result)} . wd-v1-4-vit-tagger-v2")#377MB model.onnx
                    try:
                        if gpu == True:
                            result = use_GPU_interrogation(fullpath)
                        else:
                            result = image_to_wd14_tags(fullpath,'wd-v1-4-convnextv2-tagger-v2')

                        if result == None:
                            print("nothing returned from interrogation")
                    except Exception as e:
                        print(f"interrogation failed.  {e}")
                         
                    #This gets all the tags currently in the file and creates a deduplicated list
                    mylist =[]
                    if tmp is not False:
                        for k,v in tmp.items():
                                if len(v) > 0:
                                    if isinstance(v, list):
                                        for each in v:
                                            #mylist.append(each)
                                            if ',' in each or ';' in each:
                                                    # If either a comma or semicolon is present in the value
                                                    tags = [tag.strip() for tag in re.split('[,;]', each)]  # Split the string into a list using commas and semicolons as delimiters, and remove leading/trailing spaces
                                                    mylist = mylist + tags
                                            else:
                                                mylist.append(each)

                                    else:
                                        if ',' in v or ';' in v:
                                            # If either a comma or semicolon is present in the value
                                            tags = [tag.strip() for tag in re.split('[,;]', v)]  # Split the string into a list using commas and semicolons as delimiters, and remove leading/trailing spaces
                                            mylist = mylist + tags
                                        else:
                                            mylist.append(v)
                                else:
                                    print(f"field {k} is empty")
                                

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
                    isperson = False
                    if result is not None:
                        result2 = result[1]
                        #if len(mylist) > 0:
                        if len(mylist) > 0:
                            result2 = result2 + mylist

                        #mylist = list(set(mylist))
                        #deduplicate
                        result2 = list(set(filter(None, result2))) #remove duplicates and empty values
                        result2 = [value for value in result2 if len(value) >= 3]  # Remove values under 3 characters in length


                        test = ' '.join(result2).replace('/',' ').replace("\\",' ')
                        if is_person(test):
                            #print(f" means it's is_person")
                            isperson = True
                            result2.append("ISPERSON")
                        #for res in test:
                        #    if is_person(res):
                        #        res.append(f" {res} means it's is_person")

                        if len(result2) > 0:
                            #print(str(result2))
                            tagname = 'XPKeywords'
                            #tagname = 'EXIF:XPKeywords'
                            try:
                                if filename.lower().endswith(('.png')):
                                    write_pnginfo(fullpath, result2)
                                    apply_description_keywords_tag(fullpath,result2,tag_as_processed,True)
                                elif filename.lower().endswith(('.jpg')):
                                    #modify_exif_tags(fullpath, result2, 'add',None,tagname)
                                    apply_description_keywords_tag(fullpath,result2,tag_as_processed,True)

                            except Exception as e:
                                print(f"writing tags failed.  {e}")
                        else:
                            print("stuff detected but not relevant/length 0")
                    else:
                        print(f"nothing detected for {fullpath}.  Odd")
                else:
                    print(f"{fullpath} - Tagged as processed.  Skipping")
                print(f"{fullpath} - Processing complete")
            else:
                print(f"{fullpath} - Unsupported filetype.  Not an image.")
