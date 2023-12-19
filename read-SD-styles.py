import csv

def read_style_to_list(file_path):
    data_array = []

    #with open(file_path, 'r', encoding='utf-8') as f:
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            name = row['name'].lower()
            pos_prompt = row['prompt'].lower()
            neg_prompt = row['negative_prompt'].lower()

            if not pos_prompt == '' or not neg_prompt == '':
                data_array.append(row)
            else:
                print("ignoring delimiter " + name)


    return data_array


def checkposandneg(data_dict, string1, string2):
    # Assuming the fields are named [string1example], [string2example], and [name]
    field1_name = f"prompt"
    field2_name = f"negative_prompt"
    name_field = "name"

    string1 = string1.lower()
    string2 = string2.lower()
    posprompt = []
    negprompt = []
    bothprompt = []
    both = None
    field1 = None
    field2 = None
    poscnt = 0
    negcnt = 0
    bothcnt = 0
    #print(type(data_dict))
    for entry in data_dict:
       # print(entry)
        posline = entry[field1_name].lower()
        negline = entry[field2_name].lower()
        if string1 in posline and string2 in negline and posline != '' and negline != '':
            both = entry.get(name_field)
            bothprompt.append(entry.get(name_field))
            print("both.  ",posline,negline,both)
            bothcnt +=1
        elif string2 in negline and negline != '':
            field2 =  entry.get(name_field)
            #test = negline
            negprompt.append(entry.get(name_field))
            print("field2",negline,field2)
            negcnt +=1
        elif string1 in posline and posline != '':
            field1 = entry.get(name_field)
            #test = posline
            posprompt.append(entry.get(name_field))
            print("field1",posline,field1)
            poscnt +=1

    print("Counters are both,pos,neg" ,bothcnt,poscnt,negcnt )    
    return bothprompt,posprompt,negprompt,bothcnt,poscnt,negcnt

# Example usage:
if __name__ == "__main__":
    csv_file_path = "x:/dif/stable-diffusion-webui-docker/data/config/auto/styles.csv"  # Replace with the actual path to your CSV file
    result_array = read_style_to_list(csv_file_path)

    # Display the result array
    #for row in result_array:
        #print(row)
    pospr = "Low Detail"
    negpro = "deformed"

    both,pos,neg,bthcounter,poscntr,negcntr = checkposandneg(result_array,pospr,negpro)

    #for result_dict in result_array:
    #    result = checkposandneg(result_dict, pospr, negpro)
    #    print(result)
    if both is not None:
        print("style applied both is :" + str(both) + ".  There were " + str(bthcounter) + " matches.")
    
    if pos is not None:
        print("style applied pos is :" + str(pos) + ".  There were " + str(poscntr) + " matches.")

    if neg is not None:
        print("style applied neg is :" + str(neg) + ".  There were " + str(negcntr) + " matches.")
