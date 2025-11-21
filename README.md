# Lipstick-Recommendation
**AY-AllenYe @ HDU**

Wish this project helps you. And your Star is really helpful.

This projects is aiming to provide several similar lipsticks (mainly in colors) with the user updates.

## version 1.x

### Explanation

**1.Dataset**

​	I searched from Github and got 288 items of a JSON file '\datasets\\lipstick.json' which includes attributes of 288 different lipsticks. The JSON file shows the attributes of every items such as brands, series, names, ids, and the most important one, the HEX colors. 

​	I programmed some python files to processed the HEX color, convert the HEX color to RGB, HSV, then I create a CSV file to store information the JSON file shows and the attributes I processed. Also, I made 288 JPG solid color images by the HEX value.

​	I split the dataset: 95% for training (272 items) and 5% for validation (15 items), not shuffled.

**2.Cluster**

​	I choose to use K-Means to made the cluster. 

​	To determined the number of cluster (the K in K-Means), I programmed in a 'for' loop and judge by calculating the WCSS and silhouettes score to determined the K.

​	In the program, I also visualized the curves of WCSS and silhouettes score by the increasing K, and automatically choose the best K. After the determination, the program visualized the distribution and the cluster. Moreover, the program itself created a new CSV file to save the cluster name and ID, and another new CSV file to store the dictionary of the cluster ID and related cluster name.

​	My machine suggested me set K to 5, and I did.

**3.Train**

​	I choose ResNet50 in Jittor framework (it should be okay to use Tensorflow or Pytorch, I haven't try).

​	I set batch_size to 68 (68 equals to 272 divided by 4), num_classes to 5 (the suggestion given by K-Means) and 80 epochs.

**4.Results**

​	About 1 hour to train (80 epochs). The package occupies about 9MB of space.

​	The best model (saved by the Epoch 80) scores 98.53% in Trainset, and scores 14/15 in validation set.

**5.Inference**

​	Batch or single sample is supported. 

​	To classify batch samples, users need a folder to store the JPG solid color images (PNG images is okay, but have to change one piece of code.)

​	To classify single sample, users need to type HEX color in the command.

```
the file tree
|-- main
    |-- dataset_proceed.py
    |-- infer.py
    |-- train.py
    |-- README.md
    |-- datasets
    |   |-- lipstick.json
    |-- utils
    |   |-- cluster.py
    |   |-- compute_weights.py
    |   |-- csv2dict.py
    |   |-- hex2hsv.py
    |   |-- hex2rgb.py
    |   |-- json2csv.py
    |   |-- load_dataset.py
    |   |-- logger.py
    |   |-- models.py
```

### Bash

```
cd
python dataset_proceed.py
python train.py

python infer.py
```

​	The users are probably change the path.

### Dependency

```
jittor == 1.3.8.5
numpy == 1.26.4       (In my machine these 2 wheels is tested related.)
and others.
```

### Updated Logs

#### 2025.11.20 version 1.1

**Goals (v1.1)**

    1.Optimize the file tree structure (Move and Rename several files)
    2.Fix bugs

**TODO (v1.1)**
    
    Same as TODO (v1.0)

#### 2025.11.19 version 1.0

**Goals (v1.0)**

    1.Datasets have been processed.
    2.Clusters have been determined.
    3.Classification has simply done.

**TODO (v1.0)**

    1.More datasets.(By fetching from some online shops, TaoBao and Amazon etc.)
    2.More dimension of datasets.(Not just colors, but also other attributes, glossiness and moisture etc.)
    3.Re-determine the Cluster by new-updated datasets.
    4.Find more proper and intelligent algorithm of classification.
    5.Find mor accurate models or nets to train.
    6.UI and repository, which can easily interact.
    7.Function: Recommend other items in same cluster, list its information or online shopping website.
    8.Function(Maybe): Virtual dress-on if input real-time video of human face. 