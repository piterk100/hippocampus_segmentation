import glob
import random
import os
import shutil

random.seed(42)

patients = glob.glob("NII/*")

random.shuffle(patients)

train_data = patients[:50]
val_data = patients[50:61]
test_data = patients[61:]

print(len(train_data), "+", len(val_data), "+", len(test_data))

path_train = './train_set'
path_val = './val_set'
path_test = './test_set'
path_checkpoints = './checkpoints'

os.mkdir(path_train)
os.mkdir(path_val)
os.mkdir(path_test)
os.mkdir(path_checkpoints)

for i in range(50):
  src_path = train_data[i]
  path_end = os.path.basename(os.path.normpath(src_path))
  dst_path = "train_set/" + path_end
  shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

for i in range(11):
  src_path = val_data[i]
  path_end = os.path.basename(os.path.normpath(src_path))
  dst_path = "val_set/" + path_end
  shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

for i in range(11):
  src_path = test_data[i]
  path_end = os.path.basename(os.path.normpath(src_path))
  dst_path = "test_set/" + path_end
  shutil.copytree(src_path, dst_path, dirs_exist_ok=True)