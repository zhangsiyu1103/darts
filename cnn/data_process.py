import torch
import torchvision.datasets as dataset
import numpy as np
import pickle
from PIL import Image
import os
from statistics import mean,median
from sklearn.metrics import mean_squared_error
import gist
import pandas as pd

root = "../data"
base_folder = 'cifar-10-batches-py'
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename = "cifar-10-python.tar.gz"
tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
train_list = [
    ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
    ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
    ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
    ['data_batch_4', '634d18415352ddfa80567beed471001a'],
    ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
]

test_list = [
    ['test_batch', '40351d587109b95175f43aff81a1287e'],
]
meta = {
'filename': 'batches.meta',
'key': 'label_names',
'md5': '5ff9c542aee3614f3951f8cda6e48888',
}

data = []
targets = []


for file_name, checksum in train_list:
    file_path = os.path.join(root, base_folder, file_name)
    with open(file_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        data.append(entry['data'])
        if 'labels' in entry:
            targets.extend(entry['labels'])
        else:
            targets.extend(entry['fine_labels'])

data = np.vstack(data).reshape(-1, 3, 32, 32)
data = data.transpose((0, 2, 3, 1))  # convert to HWC

sorted_data = {}
for i in range(len(data)):
    cur_target = str(targets[i])
    if cur_target in sorted_data.keys():
        sorted_data[cur_target].append(data[i])
    else:
        sorted_data[cur_target]= [data[i]]
def differentiate(img1, img2):
    desc1 = gist.extract(img1)
    desc2 = gist.extract(img2)
    mse = mean_squared_error(desc1, desc2)
    return mse


print([i for i in sorted_data.keys()])
#desc = gist.extract(data[0])
#print(data[0].shape)
#print(desc.shape)
#print(desc)
diffs={}
for key in sorted_data.keys():
    diffs[key]=[]
    cur_data = sorted_data[key]
    length = len(cur_data)
    print(key)
    print(length)
    cur_img = cur_data[0]
    for i in range(1, length):
        diff = differentiate(cur_img, cur_data[i])
        diffs[key].append(diff)
    print("median: ", median(diffs[key]))
    print("mean: ", mean(diffs[key]))
    print("min: ", min(diffs[key]))
    diffs[key] = median(diffs[key])

new_data = {"input":[],"target":[]}
for key in sorted_data.keys():
    threshold = diffs[key]*0.7
    cur_data = sorted_data[key]
    length = len(cur_data)
    cur_img = cur_data[0]
    for i in range(length):
        diff = differentiate(cur_img, cur_data[i])
        if diff < threshold:
            new_data["input"].append(cur_data[i])
            new_data["target"].append(int(key))
print(len(new_data["input"]))

df = pd.DataFrame.from_dict(new_data)
df = df.sample(frac=1).reset_index(drop=True)
new_data = df.to_dict()
torch.save(new_data, "new_cifar.pth")

#print(data[0])
#print(targets[0])

#print(data[0].shape)
#print(targets[0])


