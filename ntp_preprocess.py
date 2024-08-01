import re
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help="input data folder path")
parser.add_argument('--output_path', type=str, help="output path must be json file")
args = parser.parse_args()

folder_path = args.data_path
output_path = args.output_path

# Find all the txt files in folder_path.
txt_files = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".txt"):
            txt_files.append(os.path.join(root, file))
print(f'all files in {folder_path}: {txt_files}')
print("number of files:", len(txt_files))

json_list = []
total_of_pairs = 0
def data_clean(dataset, file_name):
    global total_of_pairs
    c = 0 # legal content
    f = 0 # content which is too long
    num_of_pair = 0 # pair{client, server}
    response_table = {}
    data_txt = dataset.split("\n\n")
    print(f"total txt length in {file_name}: {len(data_txt)}")

    for idx, data in enumerate(data_txt):
        tmp_dict = {}
        frame_number = 0

        match = re.search(r'Frame Number: (\d+)', data)
        if match == None: continue
        frame_number = int(match.group(1))
        # Make sure the data represent NTP.
        start_idx = data.find("Network Time Protocol")
        if start_idx == -1: 
            print("Data error in Frame:", int(match.group(1)))
            continue
        
        # We only need message content.
        content = ""
        content = data[start_idx:]
        first_end = content.find("\n")
        first_line = content[:first_end]
        # There has multuple types.
        private_idx = first_line.find("private")
        control_idx = first_line.find("control")
        client_idx = first_line.find("client")
        server_idx = first_line.find("server")
        
        # Preprocess content: delete redundant.
        def content_clean(contents):
            nonlocal c
            contents = re.sub(r'\[.*?\]', '', contents)
            contents = re.sub(r'^\s+', '', contents, flags=re.MULTILINE).strip()
            c += 1
            return contents
        
        # if content is too long, delete.
        if len(content) > 1000: 
            f += 1
            continue

        # Only need client and server pairs. Both single client and single server are useless.
        finial_content = ""
        if private_idx > -1 or control_idx > -1:
            continue
        elif client_idx > -1:
            response_match = re.search(r'\[Response In: (\d+)\]', content) # find relevant server response
            finial_content = content_clean(content)
            if response_match:
                response_num = int(response_match.group(1))
                response_table[response_num] = finial_content
                continue # this continue is important
            else:
                continue
        elif server_idx > -1:
            request_match = re.search(r'\[Request In: (\d+)\]', content) # find relevant client request
            finial_content = content_clean(content)
            if request_match:
                request_num = int(request_match.group(1))
                num_of_pair += 1
                if frame_number in response_table:
                    tmp_dict = {
                        "content": [response_table[frame_number], finial_content]
                    }
                else:
                    continue
            else:
                continue
        else:
            continue
                
        json_list.append(tmp_dict)
    total_of_pairs += num_of_pair
            
    print(f"Number of legal contents in {file_name}: {c}")
    print(f"Number of content which is too long in {file_name}: {f}")
    print("Number of pairs:", num_of_pair)
    print("----------")

# all txt files need to be clean
for file in txt_files:
    with open(file, "r", encoding="utf-8") as f:
        dataset = f.read()
        data_clean(dataset, file)
print(f"Total number of pairs: {total_of_pairs}")

# build a training json file
with open(output_path, "w") as json_file: 
    json.dump(json_list, json_file)

# check whether the number pairs in json file is the same as the number of preprocessing 
with open(output_path, "r", encoding="utf-8") as f:
    check = json.load(f)

q = 0
for x in check:
    if len(x["content"]) == 2: q+=1
print(f"Check!! Total number of pairs in {output_path}:", q)