import os
import csv
from PIL import Image

# create the CSV files
overview_file = open("overview.csv", "w", newline="")
overview_writer = csv.writer(overview_file)
overview_writer.writerow(["filename", "metadata"])

prompt_file = open("prompt.csv", "w", newline="")
prompt_writer = csv.writer(prompt_file)
prompt_writer.writerow(["tag:value", "count"])

negative_prompt_file = open("negative_prompt.csv", "w", newline="")
negative_prompt_writer = csv.writer(negative_prompt_file)
negative_prompt_writer.writerow(["tag:value", "count"])

# loop through all PNG files in the current directory
for filename in os.listdir("."):
    if filename.endswith(".png"):
        # open the PNG file and extract its metadata
        img = Image.open(filename)
        metadata = img.info
        
        # write the metadata to the overview CSV file
        overview_writer.writerow([filename, metadata])
        
        # process the parameters and negative_prompt tags
        if "parameters" in metadata:
            parameters = metadata["parameters"].replace("\n", "").replace("\r", "")
            parameters_dict = dict(item.split(":") for item in parameters.split(", "))
            for tag_value, count in sorted(parameters_dict.items(), key=lambda x: int(x[1]), reverse=True):
                prompt_writer.writerow([tag_value, count])
        
        if "negative_prompt" in metadata:
            negative_prompt = metadata["negative_prompt"].replace("\n", "").replace("\r", "")
            negative_prompt_dict = dict(item.split(":") for item in negative_prompt.split(", "))
            for tag_value, count in sorted(negative_prompt_dict.items(), key=lambda x: int(x[1]), reverse=True):
                negative_prompt_writer.writerow([tag_value, count])
                
# close the CSV files
overview_file.close()
prompt_file.close()
negative_prompt_file.close()
