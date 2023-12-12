import csv
import glob
import png
from collections import Counter

# Define the keys that contain comma-separated values
array_keys = ['prompt', 'negative_prompt']

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
                else:
                    metadata[key] = value

    # Write the metadata to a CSV file
    with open(f'{filename}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in metadata.items():
            if key in array_keys:
                # Count the occurrences of values for array keys
                counter = Counter(value)
                # Write the counts to a separate CSV file
                with open(f'{key}.csv', 'a', newline='') as countfile:
                    countwriter = csv.writer(countfile)
                    for item, count in counter.items():
                        countwriter.writerow([filename, item, count])
            else:
                writer.writerow([key, value])
