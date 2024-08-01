import argparse
import random
import json
import torch
import os
import warnings
import evaluate
import time
import numpy as np
from datasets import Dataset, DatasetDict
from nltk.tokenize import word_tokenize
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, AddedToken

def dns_preprocessing(dataset):
    random.seed(42)
    contents = []
    query = []
    answer = []
    q_set = set()
    a_set = set()
    c = 0
    repeat_content = 0
    empty_content = 0
    random.shuffle(dataset)
    for data in dataset:
        content = data["content"]
        if len(content) == 2:
            c += 1
            que = content[0]
            ans = content[1]
            # do not need empty and repeat data
            if que == "" or ans == "": 
                empty_content += 1
                continue
            if que in q_set and ans in a_set:
                repeat_content += 1
                continue
            q_set.add(que)
            a_set.add(ans)
            
            query.append(que)
            answer.append(ans)
        else:
            continue
    print("-----Test data information-----")
    print("Number of data:", c)
    print("Number of repeat:", repeat_content)
    print("Number of empty:", empty_content)

    new_data = {
        "query": query,
        "response": answer
    }
    new_dataset = Dataset.from_dict(new_data)
    return new_dataset

#
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help="the path for testing data, must be json file")
parser.add_argument('--output_dir', type=str, help="the path for saving the predicted txt")
parser.add_argument('--model_dir', type=str, help="the path for evaluating model")
parser.add_argument('--max_length', type=int, help="the max length for generated result")
args = parser.parse_args()

cur_path = os.getcwd()
data_dir = cur_path + args.data_dir
output_dir = cur_path + args.output_dir
model_dir = cur_path + args.model_dir
max_length = args.max_length

# check the gpu
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)

    print(f"GPU: {gpu_name} work!!")
else:
    print("No GPU can use!")

# load training data
with open(data_dir, encoding="utf-8") as f:
    lines = json.load(f)
    
dataset = dns_preprocessing(lines)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, train_size + val_size))
test_dataset = dataset.select(range(train_size + val_size, len(dataset)))

# Create a DatasetDict with the training and testing splits
dataset_dict = DatasetDict({"train": train_dataset, "val": test_dataset, "test": test_dataset})
print(dataset_dict)
print()

# tokenizer setting
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_tokens(AddedToken("\n", normalized=False))
finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# set reference
ref_bleu = []
ref = []
n = len(dataset_dict["test"]["response"])
for res in dataset_dict["test"]["response"]:
    tmp = [res]
    ref_bleu.append(tmp)
    ref.append(res)
print(f'Total evaluation numbers: {n}.')
print()

# evaluation
start_time = time.time()
print("Evaluation start!!")
pred = []
for test in dataset_dict["test"]["query"]:
    inp = "Please answer to this question: " + test
    inputs = tokenizer(inp, return_tensors="pt")
    outputs = finetuned_model.generate(**inputs, max_length=max_length)
    answer = tokenizer.decode(outputs[0])
    answer = answer[6:-4]
    pred.append(answer)
end_time = time.time()
execution_time = end_time - start_time

print("Total evaluation time：{} sec.".format(execution_time))

# save the generated result
with open(output_dir, 'w', encoding='utf-8') as file:
    for a in pred:
        file.write(a + '\n*******************\n')

# bleu
bleu = evaluate.load("bleu")
results = bleu.compute(predictions=pred, references=ref_bleu, tokenizer=word_tokenize)
print("-----------Bleu-----------")
print(results)
print("--------------------------")

# rouge
rouge = evaluate.load('rouge')
results = rouge.compute(predictions=pred, references=ref, use_aggregator=True)
print("-----------Rouge-----------")
print(results)
print("---------------------------")

# bertscore
bertscore = evaluate.load("bertscore")
results = bertscore.compute(predictions=pred, references=ref, model_type="facebook/bart-large")
precision_values = results['precision']
recall_values = results['recall']
f1_values = results['f1']

average_precision = sum(precision_values) / len(precision_values)
average_recall = sum(recall_values) / len(recall_values)
average_f1 = sum(f1_values) / len(f1_values)

print("Average Precision:", average_precision)
print("Average Recall:", average_recall)
print("Average F1:", average_f1)
print("---------------------------")
print("Min Precision:", min(precision_values))
print("Min Recall:", min(recall_values))
print("Min F1:", min(f1_values))
print("---------------------------")
print("Max Precision:", max(precision_values))
print("Max Recall:", max(recall_values))
print("Max F1:", max(f1_values))