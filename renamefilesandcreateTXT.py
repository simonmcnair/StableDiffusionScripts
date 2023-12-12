import os
from PIL import Image

for filename in os.listdir("."):
    if filename.endswith(".png"):
        with Image.open(filename) as img:
            parameter = img.info.get("parameters")
            if parameter is not None:
                output_filename = os.path.splitext(filename)[0] + ".txt"
                with open(output_filename, "w") as output_file:
                    output_file.write(parameter)
