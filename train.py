import argparse
import random
import json
import torch
import os
import warnings
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, AddedToken
from datasets import concatenate_datasets

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
    print("-----Input data information-----")
    print("Number of data:", c)
    print("Number of repeat:", repeat_content)
    print("Number of empty:", empty_content)

    new_data = {
        "query": query,
        "response": answer
    }
    new_dataset = Dataset.from_dict(new_data)
    return new_dataset


def setting_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Original tokenizer length in flan-T5:", len(tokenizer))

    tokenizer.add_tokens(AddedToken("\n", normalized=False))
    print("New tokenizer length in flan-T5:", len(tokenizer))
    # The maximum total input sequence length after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    prefix = "Please answer this question: "
    tokenized_inputs = concatenate_datasets([dataset_dict["train"], dataset_dict["val"], dataset_dict["test"]]).map(lambda x: tokenizer(x["query"], return_overflowing_tokens=True, truncation=True, max_length=1500), batched=True, remove_columns=["query", "response"])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    # min_source_length = min([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source token length: {max_source_length}")
    # print(f"Min source length: {min_source_length}")

    # The maximum total sequence length for target text after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset_dict["train"], dataset_dict["val"], dataset_dict["test"]]).map(lambda x: tokenizer(x["response"], return_overflowing_tokens=True, truncation=True, max_length=1500), batched=True, remove_columns=["query", "response"])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    # min_target_length = min([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target token length: {max_target_length}")
    # print(f"Min target length: {min_target_length}")
    print("-------------------------------------------------")
    return (tokenizer, max_source_length, max_target_length)


def tokenize(element):
    prefix = "Please answer this question: "
    inputs = [prefix + doc for doc in element["query"]]
    outputs = tokenizer(
        inputs,
        truncation=True,
        max_length=max_source_length + 6, # 6 is the token length of prefix
    )
    labels = tokenizer(
        text_target=element["response"],
        truncation=True,
        max_length=max_target_length,
    )
    
    outputs["labels"] = labels["input_ids"]
    return outputs

#
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help="path for training data, must be json file")
parser.add_argument('--output_dir', type=str, help="the path that save the model")
parser.add_argument('--batch_size', type=int, help="set the batch size")
parser.add_argument('--epochs', type=int, help="set the epochs")
parser.add_argument('--eval_steps', type=int, default=500, help="set the evaluation steps")
parser.add_argument('-gas', '--gradient_accumulation_steps', type=int, default=2, help="set the gradient accumulation steps")
parser.add_argument('--warmup_steps', type=int, default=600, help="set the warmup steps")
parser.add_argument('--precision', type=str, default="bf16", help="set the precision of model, make sure your gpu supports it")
parser.add_argument('--training_type', type=int, choices=[1, 2], default=1, help="set training type, 1 is base/mix model, 2 is update model")
parser.add_argument('--update_dir', type=str, default=None, help="path for the model which need to be updated")
args = parser.parse_args()

cur_path = os.getcwd()
data_dir = cur_path + args.data_dir
output_dir = cur_path + args.output_dir
batch_size = args.batch_size
epochs = args.epochs
eval_steps = args.eval_steps
gas = args.gradient_accumulation_steps
warmup = args.warmup_steps
precision = args.precision
training_type = args.training_type
update_dir = args.update_dir
# print(vars(args))

# check the gpu
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)

    print(f"GPU: {gpu_name} work!!")
else:
    print("CUDA can not use")

# load training data
with open(data_dir, encoding="utf-8") as f:
    lines = json.load(f)

# check output dir
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    
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
print("-----Tokenizer setting-----")

# tokenizer setting
model_name = "google/flan-t5-small"
tokenizer, max_source_length, max_target_length = setting_tokenizer(model_name)
# select the model
if training_type == 1:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
elif training_type == 2: # update model
    last_checkpoint = update_dir
    model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint)
else:
    print("Please select the training type, 1 is base or mix model, 2 is update model.")
    raise SystemExit
model.resize_token_embeddings(len(tokenizer))
print("-------------------------------------------------")

# tokenized the training data
tokenized_datasets = dataset_dict.map(
    tokenize, 
    batched=True,
    remove_columns=dataset_dict["train"].column_names
)
print(tokenized_datasets)
# check the precision type
check_bf16 = True
check_fp16 = False
check_tf32 = False
if precision == "fp16": 
    check_bf16 = False
    check_fp16 = True
elif precision == "tf32": 
    check_bf16 = False
    check_tf32 = True
    
# parameter setting
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id = -100)
args = Seq2SeqTrainingArguments(
    output_dir = output_dir,
    do_train = True,
    do_eval = True,
    bf16 = check_bf16,
    fp16 = check_fp16,
    tf32 = check_tf32,
    load_best_model_at_end=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="steps",
    
    gradient_accumulation_steps=gas,
    num_train_epochs=epochs,
    weight_decay=0.01,
    warmup_steps=warmup,
    eval_steps=eval_steps,
    lr_scheduler_type="cosine",
    learning_rate=3e-4,
    save_steps=eval_steps,
    save_total_limit = 2,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
)

# train
trainer.train()
trainer.save_model()