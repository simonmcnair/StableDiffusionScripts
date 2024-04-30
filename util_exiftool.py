import utils
from exiftool import ExifToolHelper
import os
from PIL import PngImagePlugin, Image
from PIL.ExifTags import TAGS
import re

logger = utils.get_logger(__name__ + '.log',__name__ + '_error.log')

#logger = utils.getLogger(__name__)

et = ExifToolHelper(logger=logger,common_args=['-G', '-n','-a','-P',"-overwrite_original",'-m'])

#et.terminate()

def get_metadata(filetoproc):
    res = et.get_metadata(files=filetoproc)
    return res


def readtags(filetoproc,taglist):
    res = et.get_tags(files=filetoproc, tags=taglist)
    return res

def set_tags(filetoproc,res):
    et.set_tags(filetoproc, tags=res,params=["-m","-P", "-overwrite_original"])
    logger.info(f"{et.last_stdout}")
    if '1 image files updated' not in et.last_stdout:
        logger.error(f"Error !!! {filetoproc}. {res} {et.last_stdout}")

@utils.timing_decorator
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

@utils.timing_decorator
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

def rewrite_image_file(filetoproc):
    et.execute(*["-m","-P","-all=","-tagsfromfile","@","-all", "-overwrite_original", filetoproc])


def exiftool_get_tags(filetoproc,taglist):

    exiftaglist =  et.get_tags(files=filetoproc, tags=taglist,params=["-a","-g1","-s"])

    logger.info(f"{et.last_stdout}")
    return exiftaglist