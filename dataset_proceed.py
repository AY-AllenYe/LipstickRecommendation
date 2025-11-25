# dataset_proceed.py

### input: HEX
### Future work: input image to fetch HEX automatically.


import os
from utils.json2csv import json_to_csv
from utils.cluster import cluster_kmeans
import sys
import datetime
import pandas as pd
from PIL import Image

from utils.logger import Logger

sys.stdout = Logger()

dataset_dir = 'datasets'

json_file = os.path.join(dataset_dir, 'lipstick.json')
csv_origin_file = os.path.join(dataset_dir, 'lipstick.csv')
csv_cluster_file = os.path.join(dataset_dir, 'lipstick_clusters.csv')
csv_cluster_dict_file = os.path.join(dataset_dir, 'dict.csv')

train_dir = os.path.join(dataset_dir, 'TrainSet')
os.makedirs(train_dir, exist_ok=True)
color2jpg_images_dir = os.path.join(train_dir, 'images')
color2jpg_labels_dir = os.path.join(train_dir, 'labels')
os.makedirs(color2jpg_images_dir, exist_ok=True)
os.makedirs(color2jpg_labels_dir, exist_ok=True)
label_file_path = os.path.join(color2jpg_labels_dir, 'trainval.txt')
train_label_file_path = os.path.join(color2jpg_labels_dir, 'train.txt')
val_label_file_path = os.path.join(color2jpg_labels_dir, 'val.txt')

TrainSet_percent = 0.95

if not os.path.exists(csv_origin_file):
    json_to_csv(json_file, csv_origin_file)
else:
    print("Original CSV file has been already created.")
    
if not os.path.exists(csv_cluster_file):
    cluster_kmeans(csv_origin_file, csv_cluster_file, csv_cluster_dict_file)
    # json_to_csv(json_file, csv_origin_file)
    
else:
    print("Cluster CSV file has been already created.")

df = pd.read_csv(csv_cluster_file)


with open(label_file_path, 'w', encoding='utf-8') as label_file:
    for _, row in df.iterrows():
        num_id = row['num_id']
        r = row['R']
        g = row['G']
        b = row['B']
        cluster = row['cluster']
        
        img = Image.new('RGB', (100, 100), (r, g, b))
        
        img_filename = f'{num_id}.jpg'
        img.save(os.path.join(color2jpg_images_dir, img_filename))
        
        label_file.write(f'{img_filename} {cluster}\n')

with open(label_file_path, 'r', encoding='utf-8') as label_file:
    lines = label_file.readlines()
    split_param = int(len(lines) * TrainSet_percent)

    lines_train = lines[:split_param]
    lines_val = lines[split_param:]

    with open(train_label_file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines_train)

    with open(val_label_file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines_val)