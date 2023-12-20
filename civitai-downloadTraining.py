import requests
import time
import os

def write_to_log(log_file, message):
    print(message)
    try:
        with open(log_file, 'a', encoding='utf-8') as file:
            file.write(message + '\n')
    except Exception as e:
        print(f"Error writing to the log file: {e}")

def get_models():

    # Initialize the first page
    page = 1

    while True:
        # Make API request for the current page
        headers = {'Content-Type': 'application/json'}
        params = {'page': page}

        while True:
            try:
                response = requests.get('https://civitai.com/api/v1/models?limit=100&types=LORA', headers=headers, params=params)
            except Exception as e:
                 write_to_log(logfile_path, "Error " + str(e))

            if "be back in a few minutes" in str(response.content):
                print("error.  Site down")
                exit()
                
            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception as e:
                    write_to_log(logfile_path, "Error " + str(e))
                break
            else:
                 time.sleep(5)
                 write_to_log(logfile_path, "status code: " + str(response.status_code) + " " + response.reason)
 
        # Check if there are models in the response
        if 'items' in data:
            # Extract 'id' field from each model and add it to the list
            write_to_log(logfile_path, "totalcnt = " + str(data['metadata'].get('totalItems')))
            write_to_log(logfile_path, "page = " + str(data['metadata'].get('currentPage')) + " of " + str(data['metadata'].get('totalPages')))
            for each in data['items']:
                id = each.get('id')
                name = each.get('name')
                write_to_log(logfile_path, "processing " + str(id) + f" ({name})")

 
                for each1 in each['modelVersions']:
                    model = each1.get('name')
                        
                    for file in each1['files']:
                            
                        if file.get('type') =="Training Data":
                            
                            write_to_log(successfile_path, "found training data")
                            
                            downloadurl = file.get('downloadUrl')
                            downloadfilename = file.get('name')
                            download_fullpath = os.path.join(download_to,downloadfilename)        

                            if not os.path.exists(download_fullpath):
                                write_to_log(logfile_path, "downloading " + downloadurl + ".  Filename: " + downloadfilename + ". size: " + str(file.get('sizeKB')))
                                try:
                                    response = requests.get(downloadurl)
                                except Exception as e:
                                    write_to_log(logfile_path, "Error " + str(e))


                                if response.status_code == 200:
                                    with open(download_fullpath, 'wb') as file2:
                                        file2.write(response.content)
                                    write_to_log(logfile_path, f"File downloaded successfully to {download_fullpath}")

                                    write_to_log(successfile_path, f"{model},{downloadurl},{downloadfilename}")


                                    # WRITE STUFF HERE
                                else:
                                    write_to_log(logfile_path, f"Failed to download file. Status code: {response.status_code}")
                            else:
                                write_to_log(logfile_path, "file already exists")
                        else:
                            write_to_log(logfile_path, "file type is" + f": {file.get('type')}.  Filename: {file.get('name')}")

            # Check if there are more pages
            if data['metadata'].get('currentPage') == (data['metadata'].get('totalPages')):
                break 
            else:
                page += 1
        else:
            break

download_to = 'Z:/Pics/stable-diffusion/Training & not SD/training by other/'

logfile_path = os.path.join(download_to,'logfile.log')
successfile_path = os.path.join(download_to,'successfile.log')
get_models()
