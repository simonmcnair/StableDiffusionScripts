import collections
import csv
import glob
import png

metadata_counts = collections.defaultdict(collections.Counter)

for filename in glob.glob('*.png'):
    with open(filename, 'rb') as f:
        reader = png.Reader(file=f)
        metadata = {}
        for chunk in reader.chunks():
            if chunk[0] == b'tEXt':
                key, value = chunk[1].decode('utf-8').split('\0')
                metadata[key] = value
        for key, value in metadata.items():
            metadata_counts[key][value] += 1

with open('metadata_counts.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Key', 'Value', 'Count'])
    for key, value_counts in sorted(metadata_counts.items(), key=lambda x: sum(x[1].values()), reverse=True):
        for value, count in value_counts.most_common():
            writer.writerow([key, value, count])
