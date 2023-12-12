import os
import shutil
import re
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
from PIL.ExifTags import Base
import exifread

from datetime import datetime

def sanitise_filename(filename):
    # Remove any characters that are not allowed in filenames
    filename = re.sub(r'[^\w\s\-_\.]', '', filename).strip().lower()
    # Replace any spaces with underscores
    filename = re.sub(r'\s+', '_', filename)
    # Remove any consecutive underscores
    filename = re.sub(r'_{2,}', '_', filename)
    # Return the sanitized filename
    basefilename, extension = os.path.splitext(filename)
    if len(basefilename) > 200:
        filename = basefilename[:200] + extension
    #filename = filename.lower()
    return filename


def get_jpeg_comments(filename):
    with Image.open(filename) as img:
        comments = img.info.get('comments', None)
    return comments

def copy_file_modification_time(filea, fileb):
    # Get the modification time of filea
    mtime = os.path.getmtime(filea)
    
    # Set the modification time of fileb to match filea
    shutil.copystat(filea, fileb)
    os.utime(fileb, (os.path.getatime(fileb), mtime))


def get_jpeg_exif_comments(filename):
    try:
        with Image.open(filename) as img:
            exif_data = img._getexif()
            user_comment = exif_data.get(37510)
            com = user_comment.decode('utf-8').rstrip('\x00')
            com = com.replace('\x00','').replace('UNICODE','')
    except:
        return None
    return com
    

def get_sanitized_download_time(filepath):
    # Get the modification time of the file
    mtime = os.path.getmtime(filepath)
    # Convert the modification time to a datetime object
    dt = datetime.fromtimestamp(mtime)
    # Format the datetime as a string in the format YYYY-MM-DD_HH-MM-SS
    dt_string = dt.strftime("%Y-%m-%d_%H-%M-%S")
    # Replace any spaces or colons with underscores
    sanitized_dt_string = dt_string.replace(" ", "_").replace(":", "_")
    # Return the sanitized datetime string
    return sanitized_dt_string


def move_to_subfolder(path, subfolder):
    # Check if the path is a directory or a file
    if os.path.isdir(path):
        # Create the subfolder if it doesn't exist
        subfolder_path = os.path.join(path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Move all files in the directory to the subfolder
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                base_name, ext = os.path.splitext(file)
                dest_file = os.path.join(subfolder_path, file)
                count = 1
                while os.path.exists(dest_file):
                    new_name = f"{base_name}_{count}{ext}"
                    dest_file = os.path.join(subfolder_path, new_name)
                    count += 1
                shutil.move(os.path.join(path, file), dest_file)

    elif os.path.isfile(path):
        # Create the subfolder if it doesn't exist
        subfolder_path = os.path.join(os.path.dirname(path), subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Move the file to the subfolder
        base_name, ext = os.path.splitext(os.path.basename(path))
        dest_file = os.path.join(subfolder_path, os.path.basename(path))
        count = 1
        while os.path.exists(dest_file):
            new_name = f"{base_name}_{count}{ext}"
            dest_file = os.path.join(subfolder_path, new_name)
            count += 1
        shutil.move(path, dest_file)

def extract_text_after(list_obj, text):
    for element in list_obj:
        if text in element:
            return element.split(text)[-1].strip()
    return None

def extract_text_after2(list_obj, text):
    for element in list_obj:
        if element.strip().lower().startswith(text.lower() + ":"):
            return element.split(":")[-1].strip()
    return None

def getmodel(mystr):
    test = parameter.split(",")
    try:
        res = extract_text_after2(test,"Model")
    except:
        return None

    return res
       

for filename in os.listdir("."):
    badfile = False
    changeto = get_sanitized_download_time(filename)

    if filename.endswith(".png"):
        with Image.open(filename) as img:
            try:
                parameter = img.info.get("parameters")
                if parameter is not None:
                    print(filename + " has metadata.")
                    model = getmodel(parameter)
                else:
                    print("PNG with no metadata")
                    badfile = True
            except:
                badfile = True
    elif filename.endswith(".jpeg") or filename.endswith(".jpg"):
        print("jpeg.  Extracting comments from: " + filename)
       # retval = get_jpeg_comments(filename)
       # if retval is None:
       #     print("no comments")
       # else:
       #     print (retval) 
        try:  
            parameter = get_jpeg_exif_comments(filename)
            model = getmodel(parameter)
           # input()
        except Exception as e:
            print(str(e))
            badfile = True
    else:
        print("Ignoring unsupported filetype: " + filename)
        continue

    if badfile==True:
        print(filename + " has no metadata.  Moving to nometa subdirectory")
        move_to_subfolder(filename,"nometa")
    if badfile==False:

        if model is None:
            print("No Model specified in " + filename)
            model = "No_model_specified_"
        else:
            output_filename = sanitise_filename(model + "_" + changeto + ".txt")
            print("found model: " + model + ".  Outputting parameters to " + output_filename)
            
            try:
                with open(output_filename, "w", encoding="utf-8") as output_file:
                    print("Trying to write metadata to " + output_filename)
                    output_file.write(parameter)
                    #output_file.write(parameter)
                copy_file_modification_time(filename,output_filename)
                
            except:
                output_filename = sanitise_filename("complexmodelfield_" + changeto + ".txt")
                print("shouldn't get here " + output_filename)
                with open(output_filename, "w", encoding="utf-8") as output_file:
                    output_file.write(parameter)

        try:
            moveto = sanitise_filename(model + "_" + changeto + ".png")
            if os.path.exists(model+ "_" + filename):
                print("File already exists")                
            else:
                print(filename + " metadata extracted.  Moving to " + moveto)
                shutil.move(filename, moveto)
        except Exception as e:
            print("File move failed with error " + str(e))

 
