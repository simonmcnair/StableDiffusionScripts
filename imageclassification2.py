__author__='Simon McNair'

import os,subprocess
import util_exiftool
import utils
import inference

install_requirements = False
def setup():
    install_cmds = [
        ['pip', 'install', 'pyexiftool'],
        ['pip', 'install', 'clip-interrogator'],
        ['pip', 'install', 'onnxruntime'],
        ['pip', 'install', 'pillow'],
        ['pip', 'install', 'numpy'],
        ['pip', 'install', 'huggingface_hub'],
        ['pip', 'install', 'pandas'],
        ['pip', 'install', 'opencv-python'],
    ]
    for cmd in install_cmds:
        print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))

if install_requirements:
        setup()

#python3.11 -m venv venv
#source ./venv/bin/activate

# install torch with GPU support for example:
#pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
# install clip-interrogator
#pip install clip-interrogator
#pip install clip-interrogator==0.5.4
# or for very latest WIP with BLIP2 support
#pip install clip-interrogator==0.6.0
#pip install pyexiftool
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
#importing libraries
import re
#import glob
from PIL.ExifTags import TAGS
from PIL import Image, ImageTk
#below for pngs
#pip install piexif
#import piexif
from PIL import PngImagePlugin, Image

#import piexif.helper
import re
#import threading
from pathlib import Path
import tkinter as tk

def check_gpu_present():
    try:
        # Run nvidia-smi command
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        
        # Check if GPU information is present in the output
        if 'No devices found' in output:
            return False
        else:
            return True
    except FileNotFoundError:
        # nvidia-smi command not found, so no GPU present
        return False

@utils.timing_decorator
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

@utils.timing_decorator
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

@utils.timing_decorator
def low_vram():
    if torch.cuda.is_available():
        vram_total_mb = torch.cuda.get_device_properties('cuda').total_memory / (1024**2)
        vram_info = f"GPU VRAM: **{vram_total_mb:.2f}MB**"
        if vram_total_mb< 8:
            vram_info += "<br>Using low VRAM configuration"
            logger.info(f"{vram_info}")
    if vram_total_mb <= '4': return False 
    return True

@utils.timing_decorator
def return_vram():
    if torch.cuda.is_available():
        vram_total_mb = torch.cuda.get_device_properties('cuda').total_memory / (1024**2)
    return vram_total_mb


@utils.timing_decorator
def prepend_string_to_filename(fullpath, prefix):
    # Split the full path into directory and filename
    directory, filename = os.path.split(fullpath)

    # Prepend the prefix to the filename
    new_filename = f"{prefix}{filename}"

    # Join the directory and the new filename to get the updated full path
    new_fullpath = os.path.join(directory, new_filename)

    return new_fullpath


@utils.timing_decorator
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
            read_tags = util_exiftool.readtags(filetoproc, taglist)
            for d in read_tags:
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

@utils.timing_decorator
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

@utils.timing_decorator
def tidy_tags(lst):
    #return [item.replace("'", "").replace('"', "").replace("{", "").replace("}", "") for item in lst]

    cleaned_lst = [item.replace("'", "").replace('"', "").replace("{", "").replace("}", "") for item in lst]
    if cleaned_lst != lst:
        return cleaned_lst
    else:
        return False
    
@utils.timing_decorator
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


@utils.timing_decorator
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

@utils.log_function_call
@utils.timing_decorator
def apply_description_keywords_tag(filetoproc,valuetoinsert=None,markasprocessed=False,dedupe=False):
    res = {}
    taglist =[  #"IFD0:XPKeywords",#list
                #"IPTC:Keywords",#list
                #"XMP:TagsList",#list
                #"XMP-digiKam:TagsList",#list
                #"XMP-dc:Subject",
                "MWG:Keywords",
                #"XMP-lr:HierarchicalSubject",#list
                #"XMP:HierarchicalSubject",#list
                #"XMP-acdsee:Categories",#string
                #"XMP-mediapro:CatalogSets",#list
                #"XMP:CatalogSets",#list
                #"XMP:LastKeywordIPTC",#list
                #"XMP:LastKeywordXMP",#list
                #"XMP-microsoft:LastKeywordIPTC",
                #"XMP-microsoft:LastKeywordXMP",
                #"MicrosoftPhoto:LastKeywordIPTC",
                #"EXIF:XPKeywords", #string
                #"XMP:Subject"
                ]#string
    
    stringlist =[  
                #"XMP:Categories",#string
                #"EXIF:XPKeywords", #string
                #"IFD0:XPKeywords",
                #"XMP:Subject"
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
        test2 =  util_exiftool.get_metadata(filetoproc)
        exiftaglist =  util_exiftool.exiftool_get_tags(filetoproc, taglist)
        test = exiftaglist[0]
        #logger.info(f"get_tags output: {et.last_stdout}")
    except Exception as e:
        logger.error(f"Error apply_description_keywords_tag get metadata {e}")

 #   if '232.jpg' in filetoproc.lower():
 #       print("test")

    #lengthoftaglist = len(taglist)
    
#    if (len(test) < lengthoftaglist and markasprocessed) or (len(test) <(lengthoftaglist-1) and not markasprocessed): #should be 9 returned.  MY 8 and SourceFile
#        logger.info("not enough tags defined in image")
    for each in taglist:
        if each not in test and each != 'XMP:tagged':
            if (each not in stringlist):
                logger.info(f"tag {each} does not exist in metadata for {filetoproc}")
                res[each] = list(set([value for value in keywordlist if value]))
                process[each] += len(res[each])
            elif (each in stringlist) :
                #final = list(copyofkeywordlist)
                logger.info(f"Tags ({keywordlist}) need adding to {each}")
                res[each] = seperatorstr.join(keywordlist)
#           print("added all tags to blank")
            forcewrite = True

    if keywordlist != None:
        try:
            for d in exiftaglist:
                for k, v in d.items():
                    logger.info(f"{type(v)}: {k} = {v}")
                    allkeywordsincpotentialdupes = []    
                    copyofkeywordlist = []
                    #tags2 = set()
                    tags2 = []
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
                                    forcetag = True
                                else:
                                    if len(line) >2:
                                        #tags2.add(str(line))
                                        tags2.append(str(line))
                        else:
                            if isinstance(v, (int, float)):
                                forcewrite = True
                                continue
                            v = str(v).replace("|","/")
                            logger.info(f"Not a list:{type(v)} {k}.. {v}")

                            if ',' in v and seperatorstr != ',':
                                forcetag = True
                            if ';' in v and seperatorstr != ';':
                                forcetag = True

                            if ',' in v or ';' in v:
                                logger.info(f"NOT a List. {k}.  {v} needs splitting")
                                if 'Categories' in v:
                                    logger.debug("categories detected")
                                    v = v.replace('<Categories>','').replace('<Category Assigned=1>','').replace('<Category Assigned=0>','').replace('</Category>',';').replace('</Categories>','')
                                #if k not in stringlist:
                                logger.info(f"{k} is a string.  Should be a list.  forcing retag as list. {v}")
                                forcetag = True
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
                                    
                                    tags2 = [tag2.strip() for tag2 in re.split('[,;]', v)]  # Split the string into a list using commas and semicolons as delimiters, and remove leading/trailing spaces

                                else:
                                    logger.info("this should be EXIF:XPKeywords or XMP:Categories")
                                    tags2.append(str(v))
                                    #tags2.add(str(v))
                                    #seen = set()
                                    #if v not in keywordlist:
                                    #    logger.info(f"5.")
                                    #    copyofkeywordlist = [x for x in keywordlist if x not in seen and not seen.add(x)]


                                    #for val in valuetoinsert:
                                    #        if val not in v:

                        result = tidy_tags(tags2)
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
                                forcewrite = True
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
                        
                        if forcewrite == True or len(copyofkeywordlist) >0:
                            if (k in stringlist):
                                #final = list(copyofkeywordlist)
                                logger.info(f"STRING: Tags ({copyofkeywordlist}) need adding to {k}")
                                res[k] = seperatorstr.join(copyofkeywordlist)
                            elif len(copyofkeywordlist) >0 :
                                final = sorted(list(copyofkeywordlist))
                                logger.info(f"DICT: Tags ({final}) need adding to {k}")
                                res[k] = final
                            else:
                                logger.info(f"No tags need adding to {k}")
                                if k in res:
                                    del res[k]


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
            util_exiftool.rewrite_image_file(filetoproc)
#            et.set_tags(filetoproc, tags=res,params=["-v5","-m","-P", "-overwrite_original"])
            #et.execute(*["-m","-P", "-overwrite_original","-ifd0-= -iptc-= -exif-=", filetoproc])

            util_exiftool.set_tags(filetoproc, res)
            #exiftool -ifd0:imagedescription= 
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




@utils.timing_decorator
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

@utils.timing_decorator
def search_replace_case_insensitive(search_pattern, replace_text, input_string):
    # Perform case-insensitive search
    matches = re.finditer(search_pattern, input_string, flags=re.IGNORECASE)

    # Iterate through matches and replace in a case-sensitive manner
    result = input_string
    for match in matches:
        result = result[:match.start()] + replace_text + result[match.end():]

    return result       


@utils.timing_decorator
def load_image_in_thread(image_path):

    try:
        image1 = Image.open(image_path).resize((400, 300), Image.LANCZOS)
    except Exception as e:
        logger.error(f'{e}')
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
            logger.info("hi")
            logger.info(image_path)
            logger.info(str(test))
            util_exiftool.modify_exif_tags(image_path, test, 'add')

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
                result = inference.image_to_wd14_tags(image_path)
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
cpuandgpuinterrogation = False
defaultdir = '/folder/to/process'

current_os = utils.get_operating_system()

localoverridesfile = os.path.join(utils.get_script_path(__file__), "localoverridesfile_" + utils.get_script_name(__file__) + '_' + current_os + '.py')

logfilepath = os.path.join(utils.get_script_path(__file__),utils.get_script_name(__file__) + '.log')
errorlogfilepath = os.path.join(utils.get_script_path(__file__),utils.get_script_name(__file__) + '_error.log')
logger = utils.get_logger(logfilepath, errorlogfilepath)

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
    st = StanfordNERTagger (utils.get_script_path() + '/stanford-ner/english.all.3class.distsim.crf.ser.gz', utils.get_script_path() + '/stanford-ner/stanford-ner.jar')

#ci = None
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024,expandable_segments:True"

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
                #'ViT-L-14': 'ViT-L-14/openai',
                #'ViT-L-14': 'immich-app/ViT-L-14__openai',
                #'ViT-H-14': 'ViT-H-14/laion2b_s32b_b79',
                #'ViT-H-14': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
                #'wd14': 'wd14-convnext',
                #'wd14': 'saltacc/wd-1-4-anime',
                'wd' : 'SmilingWolf/wd-v1-4-vit-tagger-v2',
                #'blip-base': 'Salesforce/blip-image-captioning-base',   # 990MB
                #'blip-large': 'Salesforce/blip-image-captioning-large', # 1.9GB
                #'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',              # 15.5GB
                #'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',      # 15.77GB
                #'git-large-coco': 'microsoft/git-large-coco'           # 1.58GB
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
                            if current_os == 'Linux' and check_gpu_present():
                                print("GPU is present.")
                                if gpu:

                                    #image = Image.open(fullpath).convert('RGB')
                                    #ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
                                    #ci = Interrogator(Config(clip_model_name="Salesforce/blip-image-captioning-large"))
                                    
                                    #result = (ci.interrogate(image))
                                    #print(result)
                                    result = inference.image_to_wd14_tags(fullpath,'wd-v1-4-vit-tagger-v2')
                                    result = result[1]
                                    print(f"Interrogator 1 output was {result[1]}")

                                    if cpuandgpuinterrogation:
                                        result2 = inference.image_to_wd14_tags(fullpath,'wd-v1-4-convnextv2-tagger-v2')
                                        print(f"Interrogator 2 output was {result2[1]}")
                                        result2 = result2[1]
                                        result = result + result2
                            else:
                                print("No GPU found.")
                                result = inference.image_to_wd14_tags(fullpath,'wd-v1-4-convnextv2-tagger-v2')
                                result = result[1]

                            if result is None:
                                logger.info("nothing returned from interrogation")
                            else:
                                print(f"  interrogator output was {result}")
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
                        keywords = result   

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


