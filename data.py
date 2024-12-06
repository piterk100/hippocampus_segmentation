import glob
import random
import os
import shutil

patients = glob.glob("C:/Users/Piotr/Documents/DL/NII/*")

random.shuffle(patients)

train_data = patients[:50]
test_data = patients[50:]

print(len(train_data), "+", len(test_data))

path_train = './train_set'
path_val = './val_set'
path_checkpoints = './checkpoints'

os.mkdir(path_train)
os.mkdir(path_val)
os.mkdir(path_checkpoints)

for i in range(50):
  src_path = train_data[i]
  path_end = os.path.basename(os.path.normpath(src_path))
  dst_path = r"C:/Users/Piotr/Documents/DL/train_set/" + path_end
  shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

for i in range(22):
  src_path = test_data[i]
  path_end = os.path.basename(os.path.normpath(src_path))
  dst_path = r"C:/Users/Piotr/Documents/DL/val_set/" + path_end
  shutil.copytree(src_path, dst_path, dirs_exist_ok=True)