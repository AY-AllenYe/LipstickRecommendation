# pred.py

import os
from PIL import Image
from utils.hex2rgb import hex_to_rgb
import jittor as jt
import jittor.transform as transform
from models import get_model  # 引入你已有的模型结构定义
from tqdm import trange
from utils.csv2dict import csv_to_dict

cluster_dict_file = 'datasets/dict.csv'
lipstick_dict = csv_to_dict(cluster_dict_file)

model_path = "models/best_train_acc_model.pkl"
# model_path = "models/best_train_loss_model.pkl"
num_classes = 5
output_dir = "output"
output_txt = os.path.join(output_dir, "result.txt")

# ===== 加载模型 =====
model = get_model(num_classes=num_classes)
model.load_state_dict(jt.load(model_path))
model.eval()

# ===== 图像预处理 =====
test_transform = transform.Compose([
    # transform.Resize((224, 224)),
    # transform.ImageNormalize(mean=[0.5], std=[0.5]),
    transform.ToTensor()
])

# ===== 推理 =====
mode = -1
print("mode = 0 => Single color")
print("mode = 1 => Batch colors")
mode = input("enter the mode:")
# mode = int(input("enter the mode:"))

if mode == '0':   # Single color
    test_color_hex = input("input HEX color:(without #)")
    r, g, b = hex_to_rgb(test_color_hex)
        
    # 创建纯色图像, 保存为 JPG
    img = Image.new('RGB', (100, 100), (r, g, b))
    img_filename = f'{test_color_hex}.jpg'
    img.save(os.path.join(output_dir, img_filename))
    
    img = test_transform(img)
    img = jt.array(img)[None, ...]  # shape: [1, 3, 224, 224]
    pred = model(img)
    pred_label = int(jt.argmax(pred, dim=1)[0].item())
    print(f"Color {test_color_hex} goes to cluster {pred_label}, named {lipstick_dict[str(pred_label)]}")

    
elif mode == '1': # Batch colors
    test_dir = "datasets/TestSetA/images/test"
    
    results = []
    image_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])
    # image_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])

    for fname in trange(len(image_files)):
        img_path = os.path.join(test_dir, image_files[fname])
        img = Image.open(img_path).convert("RGB")
        img = test_transform(img)
        
        # 正确方式：添加 batch 维度
        img = jt.array(img)[None, ...]  # shape: [1, 3, 224, 224]

        pred = model(img)
        pred_label = int(jt.argmax(pred, dim=1)[0].item())

        results.append(f"{image_files[fname]} {pred_label}")
        
    print(results)
    
    # ===== 写入 TXT 文件 =====
    with open(output_txt, "w") as f:
        for line in results:
            f.write(line + "\n")
    
else:
    print("please choose mode in 0 or 1!")
    exit()
