# Thesis: Toward Generation of Application Messages with Large Language Models for Honeypot Emulation and Protocol Testing
## Complete thesis:
https://hdl.handle.net/11296/cup4e3  (If the website is not work, please use the url below)

https://drive.google.com/file/d/1Ldz57hGfsGfZKBcJIXLzu9xbMICAbbil/view?usp=sharing
## Dataset:
We collect the NTP, DNS and HTTP datasets from open sources, please refer to the thesis for details on data sources and preprocessing.\
\
We convert the pcap files into txt files, then use ntp_preprocess.py, dns_preprocess.py, http_preprocess.py to clean the dataset. Only the application messages are retained as our training data.
## Model Training:
We use Flan-T5 small(77M) as our pre-trained model. Note: the default precision is **bf16**, please ensure that your GPU supports it.

* training_type, 1: base/mix model, 2: update model
* default path is the current path.
``` shell
python train.py --data_dir your_train_data --output_dir your_save_path --batch_size 24 --epochs 10 --eval_steps 500 --gas 2 -- warmup_steps 600 --precision bf16 --training_type 1 --update_dir None
```
## Evaluation:
The result will save as txt files in output_dir. (eg.result.txt)
``` shell
python evaluation.py --data_dir your_test_data --output_dir your_save_path --model_dir your_model_path --max_length 1000
```
***

# 碩士論文: Toward Generation of Application Messages with Large Language Models for Honeypot Emulation and Protocol Testing
## 論文網址:
https://hdl.handle.net/11296/cup4e3 (若網頁無法進入，請使用下面的網址)

https://drive.google.com/file/d/1Ldz57hGfsGfZKBcJIXLzu9xbMICAbbil/view?usp=sharing
## Dataset:
我們收集NTP, DNS, HTTP做模型訓練，資料來源與資料處理的詳細說明請參考論文。\
\
使用wireshark將資料中的pcap檔轉換成txt，並使用ntp_preprocess.py, dns_preprocess.py, http_preprocess.py做資料處理，只留下application message做訓練資料。
## Model Training:
使用Flan-T5 small(77M)做訓練。注意: 預設的參數中使用**bf16**的精準度，請確認GPU是否支援。

* 訓練模式分為 1: base/mix model, 2: update model
* 路徑預設在當前的位置
``` shell
python train.py --data_dir your_train_data --output_dir your_save_path --batch_size 24 --epochs 10 --eval_steps 500 --gas 2 -- warmup_steps 600 --precision bf16 --training_type 1 --update_dir None
```
## Evaluation:
output_dir會將生成結果以txt的方式儲存(eg.result.txt)
``` shell
python evaluation.py --data_dir your_test_data --output_dir your_save_path --model_dir your_model_path --max_length 1000
```
