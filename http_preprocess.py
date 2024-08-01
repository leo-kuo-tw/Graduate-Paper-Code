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
    c = 0
    f = 0
    number_of_pairs = 0
    response_table = {}
    data_txt = dataset.split("\n\n")
    print(f"total txt length in {file_name}: {len(data_txt)}")

    for idx, data in enumerate(data_txt):
        tmp_dict = {}
        frame_number = 0

        # Make sure the data represent HTTP.
        start_idx = data.find("Hypertext Transfer Protocol")
        line_base_idx = data.find("Line-based text data")
        data_byte = data.find("Data (")
        if start_idx == -1: continue
        
        match = re.search(r'Frame Number: (\d+)', data)
        if match == None: continue
        frame_number = int(match.group(1))
        
        # We only need message content.
        content = ""
        if line_base_idx > -1:
            content = data[start_idx:line_base_idx].strip()
        elif data_byte > -1:
            content = data[start_idx:data_byte].strip()
        else:
            content = data[start_idx:]
        
        # Preprocess content: delete redundant
        def content_clean(contents):
            contents = re.sub(r'\[.*?\]', '', contents)
            contents = re.sub(r'^\s+', '', contents, flags=re.MULTILINE).strip()
            contents = re.sub(r"\\r\\n", "", contents, flags=re.MULTILINE).strip()
            contents = re.sub(r"\n\n", r"\n", contents, flags=re.MULTILINE).strip()
            contents = re.sub(r"\\n\n", r"\n", contents, flags=re.MULTILINE).strip()
            return contents
        
        # if content is too long, delete.
        if len(content) > 2000: 
            f += 1
            continue
        
        # Only need request and response pairs.
        request_idx = content.find("Request Method:")
        response_idx = content.find("Response Version:")
        if request_idx > -1 and response_idx == -1:
            response_match = re.search(r'\[Response in frame: (\d+)\]', content) # find relevant response
            finial_content = content_clean(content)
            if response_match:
                response_num = int(response_match.group(1))
                response_table[response_num] = finial_content
                continue
            else:
                continue
        elif response_idx > -1 and request_idx == -1:
            request_match = re.search(r'\[Request in frame: (\d+)\]', content) # find relevant request
            finial_content = content_clean(content)
            if request_match:
                if frame_number in response_table:
                    number_of_pairs += 1
                    tmp_dict = {
                        "content": [response_table[frame_number], finial_content]
                    }
                else: continue
            else:
                continue
        else:
            continue
        c += 1
        json_list.append(tmp_dict)
    total_of_pairs += number_of_pairs
    
    print(f"Number of legal contents in {file_name}: {c}")
    print(f"Number of content which is too long in {file_name}: {f}")
    print("Number of pairs:", number_of_pairs)
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



