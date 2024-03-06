
from PIL.ExifTags import TAGS
from PIL import IptcImagePlugin

#pip install IPTCInfo
#python -m pip install -U pyexiftool
import exiftool
with exiftool.ExifTool() as et:
    print(et.execute("-XMP:all", "test-image.jpg"))
    et.execute("-XMP-iptcExt:digitalsourcetype=https://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia", "test-image.jpg")
#exif:imagedescription = 270
#exif:usercomment = 37510
#exif:xpcomment = 40092
IptcImagePlugin.getiptcinfo()

findtagname = "xpcomment"
findnumerictag = ""
for pil_tag, pil_tag_name in TAGS.items():
    print(f"{pil_tag} {pil_tag_name}")
    if findnumerictag != "" and findnumerictag.lower() in pil_tag.lower():
        print("This is the one you want !!")
    if findtagname != "" and findtagname.lower() in pil_tag_name.lower():
        print("This is the one you want !!")
