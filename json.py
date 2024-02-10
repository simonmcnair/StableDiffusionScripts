
import os
import json

def find_filepath_by_hash(hash_value, data):
    for filepath, hashes in data.items():
        print(f"{filepath}{hashes}{hash_value}")
        for key, value in hashes.items():
            if hash_value == value:
                print(f"found hash {hash_value} in {key}. Lora is {hashes['name']}.  trained words are {hashes['trainedWords']} ")
                return hashes['name']
    return None

def getlorahashes(foldertosearch):

    res = {}
    for root, dirs, files in os.walk(foldertosearch):
        for filename in files:
            if '.civitai.info' in filename:
                fullfilepath = os.path.join(root,filename)
                with open(fullfilepath, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)

                if '.civitai.info' in filename:
                    test = json_data['files']
                    for each in test:
                        hashes = each['hashes']
                        res[fullfilepath] = {
                                'name': json_data.get('model',None).get('name',None),
                                'trainedWords': json_data.get("trainedWords", None),
                                'AutoV2': hashes.get('AutoV2', None),
                                'SHA256': hashes.get('SHA256', None),
                                'CRC32': hashes.get('CRC32', None),
                                'BLAKE3': hashes.get('BLAKE3', None)
                            }

                        print(str(hashes))

    print("Done")
    return res

res1 = {}
dir1 = 'X:/dif/stable-diffusion-webui-docker/data/embeddings'
res1.update(getlorahashes(dir1))
dir2 = 'X:/dif/stable-diffusion-webui-docker/data/models/Lora'
res1.update(getlorahashes(dir2))

thingy = find_filepath_by_hash('A0BAC10CA3',res1)
print(thingy)
      
print("test")