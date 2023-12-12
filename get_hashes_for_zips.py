import os
import csv
from collections import defaultdict
from PIL import Image

positive_prompts = defaultdict(int)
negative_prompts = defaultdict(int)
extra_prompts = defaultdict(int)

metadata_tags = ["parameters", "Steps", "Sampler", "CFG scale", "Seed", "Size", "Model hash", "Model"]

# Loop through all PNG files in the current working directory
for filename in os.listdir("."):
    if filename.endswith(".png"):
        # Open the image file and extract the metadata
        with Image.open(filename) as img:
            metadata = img.info.get("parameters", "")

        # Split the metadata string into individual prompts
        prompts = metadata.split(",")

        # Categorize each prompt as positive, negative, or extra
        category = "extra"
        for prompt in prompts:
            if "Negative prompt:" in prompt:
                category = "negative"
            elif category == "extra":
                category = "positive"

            # Strip whitespace and any colon characters from the prompt
            prompt = prompt.strip().replace(":", "")

            # Increment the count for the prompt in the appropriate category
            if prompt:
                if category == "positive":
                    positive_prompts[prompt] += 1
                elif category == "negative":
                    negative_prompts[prompt] += 1
                else:
                    extra_prompts[prompt] += 1

        # Extract additional metadata tags
        for tag in metadata_tags:
            value = img.info.get(tag, "")
            value = value.strip().replace(":", "")
            if value:
                extra_prompts[value] += 1

# Write the prompt counts to a CSV file
with open("prompt.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["prompt", "type", "count"])
    for prompt, count in sorted(positive_prompts.items(), key=lambda x: x[1], reverse=True):
        writer.writerow([prompt, "positive", count])
    for prompt, count in sorted(negative_prompts.items(), key=lambda x: x[1], reverse=True):
        writer.writerow([prompt, "negative", count])
    for prompt, count in sorted(extra_prompts.items(), key=lambda x: x[1], reverse=True):
        if prompt not in positive_prompts and prompt not in negative_prompts:
            writer.writerow([prompt, "extra", count])
