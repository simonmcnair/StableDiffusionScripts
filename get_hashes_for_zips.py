import os
import csv
from PIL import Image

positive_prompts = []
remainder_prompts = []
occurrences = {}

# Loop through all PNG files in the current working directory
for filename in os.listdir('.'):
    if os.path.isfile(filename):

        if filename.endswith('.png'):
            with Image.open(filename) as im:
                # Extract the metadata from the PNG file
                metadata = im.info
                #metadata_lines = metadata.splitlines()
                if 'parameters' in metadata:
                    for key, value in metadata.items():
                        #print(f'The value for key {key} is {value}')
                        value = value.replace("\n", "").replace("\r", "")
                        if 'Negative prompt: ' in value:
                            onlypositive = False
                            my_list = value.split('Negative prompt: ')
                            positive = my_list[0].split(",")
                            positive = [x.strip() for x in positive]
                
                            remainder = my_list[1].split(",")
                            remainder = [x.strip() for x in remainder]

                        else:
                            onlypositive = True
                            remainder = value.split(",")
                            #todo no remainder prompt

                    steps=[]
                    steps += [x for x in remainder if "Steps:" in x]
                    remainder = [x for x in remainder if "Steps:" not in x]
                    
                    sampler=[]
                    sampler += [x for x in remainder if "Sampler:" in x]
                    remainder = [x for x in remainder if "Sampler:" not in x]

                    CFGScale=[]
                    CFGScale += [x for x in remainder if "CFG scale:" in x]
                    remainder = [x for x in remainder if "CFG scale:" not in x]

                    #Seed = [x for x in remainder if "Seed:" in x]
                    Seed = []
                    Seed += [x for x in remainder if "Seed:" in x]
                    remainder = [x for x in remainder if "Seed:" not in x]

                    Size=[]
                    Size += [x for x in remainder if "Size:" in x]
                    remainder = [x for x in remainder if "Size:" not in x]

                    ModelHash=[]
                    ModelHash += [x for x in remainder if "Model hash:" in x]
                    remainder = [x for x in remainder if "Model hash:" not in x]

                    Model= [x for x in remainder if "Model:" in x]
                    remainder = [x for x in remainder if "Model:" not in x]

                    if onlypositive == True:
                        positive = remainder

                        #print ("Positive: " , end='')
                        #print(','.join(positive)) 
                        #print( "Seed: " + Seed)
                    else:
                        negative = remainder
                        #print ("Positive: " , end='')
                        #print(','.join(positive), ) 
                        #print ("Negative: " , end='')
                        #print(','.join(negative)) 
                        #print( "Seed: " + Seed)
                else:
                    print(filename + " is not a Stable diffusion file")
        else:
            print(filename + " is not a png")
                    # Extract the values from the "parameters" tag and add them to the positive prompts list





count_dict = {}
for item in lst:
    if item in count_dict:
        count_dict[item] += 1
    else:
        count_dict[item] = 1

with open("output.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["item", "count"])
    for item, count in count_dict.items():
        writer.writerow([item, count])