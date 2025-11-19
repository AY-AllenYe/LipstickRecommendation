import os
from utils.json2csv import json_to_csv
import sys
import datetime
import pandas as pd
from PIL import Image

from utils.logger import Logger

sys.stdout = Logger()

json_file = 'datasets/lipstick.json'
csv_origin_file = 'datasets/lipstick.csv'
csv_cluster_file = 'datasets/lipstick_clusters.csv'

color2jpg_images_dir = 'datasets/TrainSet/images'
color2jpg_labels_dir = 'datasets/TrainSet/labels'

if not os.path.exists(csv_origin_file):
    json_to_csv(json_file, csv_origin_file)
else:
    print("Original CSV file has been already created.")
    
if not os.path.exists(csv_cluster_file):
    import kmeans
    # json_to_csv(json_file, csv_origin_file)
    
else:
    print("Cluster CSV file has been already created.")


# 读取 CSV 文件
df = pd.read_csv(csv_cluster_file)

# 确保输出目录存在
os.makedirs(color2jpg_images_dir, exist_ok=True)
os.makedirs(color2jpg_labels_dir, exist_ok=True)

label_file_path = os.path.join(color2jpg_labels_dir, 'trainval.txt')

train_percent = 0.95
train_label_file_path = os.path.join(color2jpg_labels_dir, 'train.txt')
val_label_file_path = os.path.join(color2jpg_labels_dir, 'val.txt')

with open(label_file_path, 'w', encoding='utf-8') as label_file:
    # 遍历每一行，生成图片和标签
    for _, row in df.iterrows():
        num_id = row['num_id']
        r = row['R']
        g = row['G']
        b = row['B']
        cluster = row['cluster']
        
        # 创建纯色图像
        img = Image.new('RGB', (100, 100), (r, g, b))
        
        # 保存为 JPG
        img_filename = f'{num_id}.jpg'
        img.save(os.path.join(color2jpg_images_dir, img_filename))
        
        # 写入标签信息
        label_file.write(f'{img_filename} {cluster}\n')

with open(label_file_path, 'r', encoding='utf-8') as label_file:
    lines = label_file.readlines()
    split_index = int(len(lines) * train_percent)

    # 分割数据
    lines_train = lines[:split_index]  # 前95%
    lines_val = lines[split_index:]  # 后5%

    with open(train_label_file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines_train)

    with open(val_label_file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines_val)