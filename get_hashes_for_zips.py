import glob
import png

for filename in glob.glob('*.png'):
    with open(filename, 'rb') as f:
        reader = png.Reader(file=f)
        metadata = {}
        for chunk in reader.chunks():
            if chunk[0] == b'tEXt':
                key, value = chunk[1].decode('utf-8').split('\0')
                metadata[key] = value

    print(f'Metadata for {filename}: {metadata}')
