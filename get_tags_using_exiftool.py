#pip install PyExifTool

from pathlib import Path
import os
import util_exiftool
import platform
from util_exiftool import ExifToolHelper
import shlex

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


def find_metadata_tag(filetoproc,tag):
    with util_exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(filetoproc)
        for d in metadata:
            for key, value in d.items():
                if tag.lower() in key.lower():
                    return value
    print(f"Tag not found: {tag}")
    return None

def modify_metadata_tag(filetoproc,tag,operation='read',valuetoinsert=None,format=None):
    if operation == "set":
        if valuetoinsert is None:
            return None
        # Implement logic for create operation using 'data'
        print("Creating:")
    elif operation == "read":
        # Implement logic for read operation using 'data'
        print("Reading:")
    elif operation == "delete":
        if valuetoinsert is not None:
            return None
        # Implement logic for delete operation using 'data'
        print(f"Deleting: {tag}")
    elif operation == "add":
        if valuetoinsert is None:
            return None
        # Implement logic for delete operation using 'data'
        print(f"add {valuetoinsert} to tag {tag}")
    else:
        # Invalid or unspecified operation
        print("Invalid operation or not specified.")    
        return None
    tagname = None
    tagvalues = None
    found = False
    with util_exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(filetoproc)
        for d in metadata:
            for key, value in d.items():
                if tag.lower() in key.lower():
                    if operation == "read":
                        return value
                    found = True
                    tagname = key
                    tagvalues = value
    if operation == "read":
        print(f"tag {tagname} not found")
        return None
    if tagname == None:
        tagname = tag
    if operation == "delete":
        print(f"delete tag {tagname}")
        with util_exiftool.ExifToolHelper(common_args=['-G', '-n', '-api', 'largefilesupport=1','-overwrite_original','-P','-m','-sep',';']) as et:
            res = et.execute(f"-{tagname}=", filetoproc)
    elif operation == "set":
        print(f"set tag {tagname} to {valuetoinsert}")
        if found == True:
            if valuetoinsert == tagvalues:
                print("Already correct")
                return True
            with util_exiftool.ExifToolHelper(common_args=['-G', '-n', '-api', 'largefilesupport=1','-overwrite_original','-P','-m','-sep',';']) as et:
                res = et.execute(f"-{tagname}={valuetoinsert}", filetoproc)
    elif operation == "add":
        if found == True:
                if valuetoinsert in tagvalues:
                    print(f"Tag {valuetoinsert} already exists in {tagname}.  Values are {tagvalues}")
                    return True
                print(f"add value {valuetoinsert} to {tagname}")
                with util_exiftool.ExifToolHelper(common_args=['-G', '-n', '-api', 'largefilesupport=1','-overwrite_original','-P','-m','-sep',';']) as et:
                    res = et.execute(f"-{tagname}+={valuetoinsert}", filetoproc)
        else:
            print(f"add value {valuetoinsert} to {tagname}")
            with util_exiftool.ExifToolHelper(common_args=['-G', '-n', '-api', 'largefilesupport=1','-overwrite_original','-P','-m','-sep',';']) as et:

                try:
                    res = et.execute(f"-{tagname}+={valuetoinsert}", filetoproc)
                    if et.last_stderr != '':
                        print(et.last_stderr)
                except Exception as e:
                    print(str(e))
                    if et.last_status ==1:
                        print(et.last_stderr)
    return res
   
def apply_description_metadata_tag(filetoproc,valuetoinsert=None):
    res = {}
    taglist =[]
    taglist.append("EXIF:XPSubject")
    taglist.append("XMP:Caption")
    taglist.append("EXIF:ImageDescription")
    taglist.append("IPTC:Caption-Abstract")
    taglist.append("XMP:Description")
    taglist.append("XMP:ImageDescription")
    try:
        with ExifToolHelper() as et:
            for d in et.get_tags(files=filetoproc, tags=taglist):
                for k, v in d.items():
                    print(f"Dict: {k} = {v}")
                    if k != 'SourceFile':
                        res[k] = valuetoinsert
            et.set_tags(filetoproc, tags=res,params=["-P", "-overwrite_original"])
        return True
    except Exception as e:
        print(f"Error {e}")
        return False
def apply_description_comment_tag(filetoproc,valuetoinsert=None):
    res = {}
    taglist =[]
    taglist.append("EXIF:XPComment")
    taglist.append("XMP:UserComment")
    taglist.append("EXIF:UserComment")
    taglist.append("XMP:Notes")
    try:
        with ExifToolHelper() as et:
            for d in et.get_tags(files=filetoproc, tags=taglist):
                for k, v in d.items():
                    print(f"Dict: {k} = {v}")
                    if k != 'SourceFile':
                        res[k] = valuetoinsert
            et.set_tags(filetoproc, tags=res,params=["-P", "-overwrite_original"])
        return True
    except Exception as e:
        print(f"Error {e}")
        return False
def apply_description_title_tag(filetoproc,valuetoinsert=None):
    res = {}
    taglist =[]
    taglist.append("EXIF:XPTitle")
    taglist.append("XMP:Title")
    taglist.append("IPTC:ObjectName")
    try:
        with ExifToolHelper() as et:
            for d in et.get_tags(files=filetoproc, tags=taglist):
                for k, v in d.items():
                    print(f"Dict: {k} = {v}")
                    if k != 'SourceFile':
                        res[k] = valuetoinsert
            et.set_tags(filetoproc, tags=res,params=["-P", "-overwrite_original"])
        return True
    except Exception as e:
        print(f"Error {e}")
        return False
def apply_description_keywords_tag(filetoproc,valuetoinsert=None):
    res = {}
    taglist =[]
    taglist.append("IPTC:Keywords")#list
    taglist.append("XMP:TagsList")
    taglist.append("XMP:Hierarchicalsubject")
    taglist.append("XMP:Categories")
    taglist.append("XMP:CatalogSets")
    taglist.append("XMP:LastKeywordXMP")
    taglist.append("EXIF:XPKeywords") #string
    taglist.append("XMP:subject")
    if valuetoinsert != None:
        keywordlist = valuetoinsert.split(',')
    elif valuetoinsert == 'del':
        keywordlist = None

    try:
        with ExifToolHelper() as et:
            if keywordlist != None:
                for d in et.get_tags(files=filetoproc, tags=taglist):
                    for k, v in d.items():
                        print(f"Dict: {k} = {v}")
                        if k != 'SourceFile':
                            #if k == 'IPTC:Keywords' or k == 'XMP:LastKeywordXMP':
                                if isinstance(v, list):
                                    for line in v:
                                        #if line not in v:
                                            copyofkeywordlist = keywordlist
                                            copyofkeywordlist.append(line.strip())
                                        #else:
                                        #    print(f"{line} already in {v}")
                                    res[k] = copyofkeywordlist
                                else:
                                    if ',' in v:
                                        tags = v.split(',')
                                        for val in valuetoinsert.split(','):
                                            if val not in tags:
                                                tags.append(val.strip())
                                            else:
                                                print(f"{val} already in {tags}")
                                        res[k] = ';'.join(tags)
                                    else:
                                        res[k] = valuetoinsert
                for key, value in res.items():
                    if isinstance(value, list):
                        res[key] = list(set(value))
                et.set_tags(filetoproc, tags=res,params=["-P", "-overwrite_original"])
            else:
                deletestring = ""
                for each in taglist:
                    deletestring = deletestring + f"-{each}= "
                et.execute(deletestring, filetoproc)

            print("test")
        return True
    except Exception as e:
        print(f"Error {e}")
        return False

def get_description_keywords_tag(filetoproc):
    res = {}
    taglist =[]
    taglist.append("IPTC:Keywords")#list
    taglist.append("XMP:TagsList")
    taglist.append("XMP:Hierarchicalsubject")
    taglist.append("XMP:Categories")
    taglist.append("XMP:CatalogSets")
    taglist.append("XMP:LastKeywordXMP")
    taglist.append("EXIF:XPKeywords") #string
    taglist.append("XMP:subject")
    keywordlist = []
    import re
    try:
        with ExifToolHelper() as et:
                for d in et.get_tags(files=filetoproc, tags=taglist):
                    for k, v in d.items():
                        print(f"Dict: {k} = {v}")
                        if k != 'SourceFile':
                                if isinstance(v, list):
                                    for line in v:
                                            keywordlist.append(line.strip())
                                    res[k] = keywordlist
                                else:
                                    if ',' in v or ';' in v:
                                        # If either a comma or semicolon is present in the value
                                        tags = [tag.strip() for tag in re.split('[,;]', v)]  # Split the string into a list using commas and semicolons as delimiters, and remove leading/trailing spaces
                                        res[k] = ';'.join(tags)  # Join the list elements using semicolon as the separator and assign to the key k in the dictionary res
                                    else:
                                        res[k] = v
                for key, value in res.items():
                    if isinstance(value, list):
                        res[key] = list(set(value))
        return True
    except Exception as e:
        print(f"Error {e}")
        return False
 

def read_all_metadata(filetoproc):
    with util_exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(filetoproc)
        for d in metadata:
            for key, value in d.items():
                #print(key, value)
                if isinstance(value, (int, float)):
                    # If the value is a number, format it as a float with two decimal places
                    formatted_value = "{:0.2f}".format(value)
                elif isinstance(value,dict):
                    formatted_value = ', '.join(f"{subkey}: {subvalue}" for subkey, subvalue in value.items())
                elif isinstance(value,list):
                    formatted_value = ','.join(f"{item}" for item in value)
                    #formatted_value = ', '.join(str(value))
                else:
                    # If the value is not a number, keep it as is
                    formatted_value = value
                print("{:30.30} {}".format(key,formatted_value))
            #for key,value in d:
            #    print(key + value)

            #    print("{:20.20} {:20.20}".format(d["SourceFile"],d["EXIF:DateTimeOriginal"]))
            #    print("{:20.20} {:20.20}".format(d["SourceFile"],d["EXIF:DateTimeOriginal"]))
        return metadata



imagefile = '/path/to/test.jpg'
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

apply_description_metadata_tag(imagefile,'example description')
apply_description_comment_tag(imagefile,'example comment')
apply_description_title_tag(imagefile,'example title')
apply_description_keywords_tag(imagefile,'keyword1,keyword2,keyword3')

#print(modify_metadata_tag(imagefile,'IPTC:CatalogSets','add','bob'))
#print(modify_metadata_tag(imagefile,'XMP:Hierarchicalsubject','set','bob'))




