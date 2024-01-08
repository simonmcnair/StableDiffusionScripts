#pip install PyExifTool

from pathlib import Path
import os
import exiftool


        
def get_script_name():
    # Use os.path.basename to get the base name (script name) from the full path
    #basename = os.path.basename(path)
    return Path(__file__).stem
    #return os.path.basename(__file__)

def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))



def test(filetoproc):
    with exiftool.ExifToolHelper() as et:
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







imagefile = '/path/to/test.jpg'
localoverridesfile = os.path.join(get_script_path(), "localoverridesfile_" + get_script_name() + '.py')

if os.path.exists(localoverridesfile):
    exec(open(localoverridesfile).read())
else:
    print("No local overrides.")

test(imagefile)

