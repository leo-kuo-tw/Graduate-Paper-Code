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
json_list = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".txt"):
            txt_files.append(os.path.join(root, file))
print(f'all files in {folder_path}: {txt_files}')
print("number of files:", len(txt_files))

DNSSEC_dict = {"NSEC", "NSEC3", "RRSIG", "DNSKEY", "DS", "DLV"}
total_of_pairs = 0
def data_clean(dataset, file_name):
    global total_of_pairs
    response_table = {}
    total_type = {}
    c = 0
    tooLen = 0
    num_of_pair = 0
    data_txt = dataset.split("\n\n")
    print(f"total txt length in {file_name}: {len(data_txt)}")

    for i, d in enumerate(data_txt):
        check_dnssec = 1
        tmp_dict = {}
        tmp_type = {}
        match = re.search(r'Frame Number: (\d+)', d)
        if match == None: continue
        frame_number = int(match.group(1))
        # Make sure the data represent HTTP.
        start_idx = d.find("Domain Name System")
        if start_idx == -1: continue
        # We only need message content.
        content = d[start_idx:]

        # Do not need the type that is in DNSSEC_dict. These dict can adjust by yourself.
        type_matches = re.findall(r'Type: (.*?) \(', content, re.DOTALL)
        for match in type_matches:
            if match in DNSSEC_dict: 
                check_dnssec = 0
                continue
        if check_dnssec == 0:
            continue
        
        # Preprocess content: delete redundant.
        def content_clean(contents):
            contents = contents.replace("<", "")
            contents = contents.replace(">", "")
            contents = re.sub(r'\[.*?\]', '', contents)
            contents = re.sub(r'^\s+', '', contents, flags=re.MULTILINE).strip()
            return contents

        # if content is too long, delete.
        if len(content) > 2000: 
            tooLen += 1
            continue

        response_match = re.search(r'\[Response In: (\d+)\]', content)
        request_match = re.search(r'\[Request In: (\d+)\]', content)
        # control the number of the type, balance the training data 
        if response_match or request_match:
            for match in type_matches:
                if match not in tmp_type: tmp_type[match] = 1
        if "A" in tmp_type and "A" in total_type and total_type["A"] >= 20000: continue
        if "AAAA" in tmp_type and "AAAA" in total_type and total_type["AAAA"] >= 10000: continue
        if "SOA" in tmp_type and "SOA" in total_type and total_type["SOA"] >= 10000: continue
        if "PTR" in tmp_type and "PTR" in total_type and total_type["PTR"] >= 10000: continue
        if "NS" in tmp_type and "NS" in total_type and total_type["NS"] >= 10000: continue
        if "CNAME" in tmp_type and "CNAME" in total_type and total_type["CNAME"] >= 10000: continue
        if "OPT" in tmp_type and "OPT" in total_type and total_type["OPT"] >= 10000: continue
        if "Unused" in tmp_type: continue

        for idx, val in tmp_type.items():
            if idx not in total_type: total_type[idx] = 1
            else: total_type[idx] += 1
        
        # Only need request and response pairs.
        if response_match:
            response_num = int(response_match.group(1))
            finial_content = content_clean(content)
            if response_num in response_table: print(f"{response_num} already in table!!")
            response_table[response_num] = finial_content
            continue
        elif request_match:
            request_num = int(request_match.group(1))
            finial_content = content_clean(content)
            num_of_pair += 1
            if frame_number in response_table:
                tmp_dict = {
                    "content": [response_table[frame_number], finial_content]
                }
            else:
                continue
        else:
            continue

        c += 1
        json_list.append(tmp_dict)
    total_of_pairs += num_of_pair
            
    print(f"Number of legal contents in {file_name}: {c}")
    print(f"Number of content which is too long in {file_name}: {tooLen}")
    print("Number of pairs:", num_of_pair)
    print("======================")

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
