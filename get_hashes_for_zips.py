import csv
import glob
import png
from collections import Counter

# Define the keys that contain comma-separated values
array_keys = ['prompt', 'negative_prompt']

# Initialize counters for each array key
array_counters = {key: Counter() for key in array_keys}

# Loop over each PNG file in the working directory
for filename in glob.glob('*.png'):
    with open(filename, 'rb') as f:
        reader = png.Reader(file=f)
        metadata = {}
        for chunk in reader.chunks():
            if chunk[0] == b'tEXt':
                key, value = chunk[1].decode('utf-8').split('\0')
                if key in array_keys:
                    metadata[key] = value.split(',')
                    # Increment the counters for each value in the array
                    array_counters[key].update(metadata[key])
                else:
                    metadata[key] = value

    # Write the metadata to a CSV file
    with open(f'{filename}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in metadata.items():
            if key not in array_keys:
                writer.writerow([key, value])

# Write the final counts to CSV files
for key, counter in array_counters.items():
    with open(f'{key}.csv', 'w', newline='') as countfile:
        countwriter = csv.writer(countfile)
        for item, count in counter.items():
            countwriter.writerow([item, count])
