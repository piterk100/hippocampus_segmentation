import torch
import os
import glob
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib

def get_sub_volume(image, label,
                   orig_x = 512, orig_y = 512, orig_z = 140,
                   output_x = 128, output_y = 128, output_z = 16,
                   num_classes = 1, max_tries = 1000,
                   hipo_threshold = 0.01):

    X = None
    y = None

    tries = 0

    orig_z = label.shape[2]

    max_hipo_ratio = 0

    while tries < max_tries:
        start_x = np.random.randint(0, orig_x - output_x + 1)
        start_y = np.random.randint(0, orig_y - output_y + 1)
        start_z = np.random.randint(0, orig_z - output_z + 1)

        y = label[start_x:(start_x + output_x),
                  start_y:(start_y + output_y),
                  start_z:(start_z + output_z)]

        hipo_ratio = np.sum(y) / (output_x * output_y * output_z)

        tries += 1

        if hipo_ratio > hipo_threshold:
            X = np.copy(image[start_x: start_x + output_x,
                              start_y: start_y + output_y,
                              start_z: start_z + output_z])

            return X, y

        if hipo_ratio > max_hipo_ratio:
            max_hipo_ratio = hipo_ratio
            best_x, best_y, best_z = start_x, start_y, start_z

    X = np.copy(image[best_x: best_x + output_x,
                      best_y: best_y + output_y,
                      best_z: best_z + output_z])

    return X, y

def standardize(image):

    standardized_image = np.zeros(image.shape)

    for z in range(image.shape[2]):
        image_slice = image[:,:,z]

        centered = image_slice - np.mean(image_slice)
        centered_scaled = centered

        if np.std(centered) != 0:
            centered_scaled = centered / np.std(centered)

        standardized_image[:, :, z] = centered_scaled

    return standardized_image

class DatasetFromNii(Dataset):
  def __init__(self, image_dir, transform=None):
    super(DatasetFromNii, self).__init__()

    self.patients = glob.glob(image_dir + "*")
    self.data_len = len(self.patients)
    self.images = os.listdir(image_dir)

    self.x = []
    self.y = []
    self.z = []

    for index in range(self.data_len):
        single_image_path = self.patients[index] + "/ct.nii.gz"
        single_image_path_l = self.patients[index] + "/segmentations/Hippocampus_L.nii.gz"
        single_image_path_r = self.patients[index] + "/segmentations/Hippocampus_R.nii.gz"

        single_image_array = nib.load(single_image_path).get_fdata()
        single_image_array_l = nib.load(single_image_path_l).get_fdata()
        single_image_array_r = nib.load(single_image_path_r).get_fdata()

        single_image_array_l[single_image_array_l == 255] = 1
        single_image_array_r[single_image_array_r == 255] = 1
        single_image_array_plus = single_image_array_l + single_image_array_r

        single_image_array, single_image_array_plus = get_sub_volume(single_image_array, single_image_array_plus)
        single_image_array = standardize(single_image_array)

        #single_image_array_plus = np.where(single_image_array_plus == 1, 255, 0)

        self.x.append(torch.tensor(single_image_array, dtype=torch.float32))
        self.y.append(torch.tensor(single_image_array_plus, dtype=torch.float32))
        self.z.append(torch.tensor(single_image_array_r, dtype=torch.float32))

  def __getitem__(self, index):
    if torch.is_tensor(index):
            index = index.tolist()

    proccessed_out = {'image': self.x[index], 'mask': self.y[index]}

    return proccessed_out

  def __len__(self):
    return self.data_len