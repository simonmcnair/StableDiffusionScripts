from PIL import Image
import os
import csv

# Initialize the counters for the prompt and negative_prompt tags
prompt_counts = {}
negative_prompt_counts = {}

# Open the output CSV files for writing
with open("overview.csv", "w", newline="", encoding="utf-8") as overview_file, \
     open("prompt.csv", "w", newline="", encoding="utf-8") as prompt_file, \
     open("negative_prompt.csv", "w", newline="", encoding="utf-8") as negative_prompt_file:

    # Create CSV writers for each file
    overview_writer = csv.writer(overview_file)
    prompt_writer = csv.writer(prompt_file)
    negative_prompt_writer = csv.writer(negative_prompt_file)

    # Write the headers for the CSV files
    overview_writer.writerow(["Filename", "Metadata"])
    prompt_writer.writerow(["Tag", "Value", "Count"])
    negative_prompt_writer.writerow(["Tag", "Value", "Count"])

    # Traverse the PNG files in the current working directory
    for filename in os.listdir("."):
        if filename.endswith(".png"):
            # Open the PNG file and extract the metadata
            with Image.open(filename) as img:
                metadata = img.info

            # Write the metadata to the overview CSV file
            overview_writer.writerow([filename, metadata])

            # Process the parameters tag in the metadata
            if "parameters" in metadata:
                parameters = metadata["parameters"].replace("\n", "").replace("\r", "")
                parameter_dict = {}
                for param in parameters.split(","):
                    if ":" in param:
                        key, value = param.split(":", 1)
                        parameter_dict[key.strip()] = value.strip()
                for key, value in parameter_dict.items():
                    # Update the prompt counts
                    prompt_counts.setdefault(key, {}).setdefault(value, 0)
                    prompt_counts[key][value] += 1
                    # Write the prompt tag counts to the CSV file
                    prompt_writer.writerow([key, value, prompt_counts[key][value]])

            # Process the negative_prompt tag in the metadata
            if "negative_prompt" in metadata:
                negative_prompt = metadata["negative_prompt"].replace("\n", "").replace("\r", "")
                negative_prompt_dict = {}
                for param in negative_prompt.split(","):
                    if ":" in param:
                        key, value = param.split(":", 1)
                        negative_prompt_dict[key.strip()] = value.strip()
                for key, value in negative_prompt_dict.items():
                    # Update the negative_prompt counts
                    negative_prompt_counts.setdefault(key, {}).setdefault(value, 0)
                    negative_prompt_counts[key][value] += 1
                    # Write the negative_prompt tag counts to the CSV file
                    negative_prompt_writer.writerow([key, value, negative_prompt_counts[key][value]])
