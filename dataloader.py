import torch
import glob
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib

def get_sub_volume(image, label,
                   orig_x = 512, orig_y = 512, orig_z = 140,
                   output_x = 128, output_y = 128, output_z = 16,
                   max_tries = 1000, hipo_threshold = 0.005):

    X = None
    y = None

    tries = 0

    orig_x, orig_y, orig_z = label.shape

    output_x = min(output_x, orig_x)
    output_y = min(output_y, orig_y)
    output_z = min(output_z, orig_z)

    best_x = max(0, (orig_x - output_x) // 2)
    best_y = max(0, (orig_y - output_y) // 2)
    best_z = max(0, (orig_z - output_z) // 2)
    max_hipo_ratio = -1.0

    # jeśli w wolumenie w ogóle jest fg, wytnij patch wokół losowego voxela fg
    fg_idx = np.argwhere(label > 0)
    if fg_idx.size > 0:
        cx, cy, cz = fg_idx[np.random.randint(len(fg_idx))]
        start_x = int(np.clip(cx - output_x // 2, 0, orig_x - output_x))
        start_y = int(np.clip(cy - output_y // 2, 0, orig_y - output_y))
        start_z = int(np.clip(cz - output_z // 2, 0, orig_z - output_z))
        X = np.copy(image[start_x:start_x+output_x,
                          start_y:start_y+output_y,
                          start_z:start_z+output_z])
        y = np.copy(label[start_x:start_x+output_x,
                          start_y:start_y+output_y,
                          start_z:start_z+output_z])
        return X, y

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
            y = np.copy(label[start_x: start_x + output_x,
                              start_y: start_y + output_y,
                              start_z: start_z + output_z])  
            return X, y

        if hipo_ratio > max_hipo_ratio:
            max_hipo_ratio = hipo_ratio
            best_x, best_y, best_z = start_x, start_y, start_z

    X = np.copy(image[best_x: best_x + output_x,
                      best_y: best_y + output_y,
                      best_z: best_z + output_z])
    y = np.copy(label[best_x: best_x + output_x,
                  best_y: best_y + output_y,
                  best_z: best_z + output_z])

    return X, y

def standardize(image):
    # łagodny clip outlierów
    p2, p98 = np.percentile(image, [2, 98])
    image = np.clip(image, p2, p98)
    # z-score per-volume
    m = image.mean()
    s = image.std() + 1e-8
    return (image - m) / s


class DatasetFromNii(Dataset):
  def __init__(self, image_dir, patches_per_patient=16, mode="train"):
    super(DatasetFromNii, self).__init__()

    self.patients = glob.glob(image_dir + "*")
    self.data_len = len(self.patients)
    self.mode = mode

    self.x = []
    self.y = []

    for index in range(self.data_len):
        single_image_path = self.patients[index] + "/ct.nii.gz"
        single_image_path_l = self.patients[index] + "/segmentations/Hippocampus_L.nii.gz"
        single_image_path_r = self.patients[index] + "/segmentations/Hippocampus_R.nii.gz"

        single_image_array = nib.as_closest_canonical(nib.load(single_image_path)).get_fdata()
        single_image_array = standardize(single_image_array)
        single_image_array_l = nib.as_closest_canonical(nib.load(single_image_path_l)).get_fdata()
        single_image_array_r = nib.as_closest_canonical(nib.load(single_image_path_r)).get_fdata()

        single_image_array_plus = ((single_image_array_l > 0) | (single_image_array_r > 0)).astype(np.float32)

        if self.mode == "train":
            # --- LOSOWE PATCHY (dotychczasowe zachowanie) ---
            for _ in range(patches_per_patient):
                patch_img, patch_mask = get_sub_volume(single_image_array, single_image_array_plus)
                self.x.append(torch.tensor(patch_img, dtype=torch.float32))
                self.y.append(torch.tensor(patch_mask, dtype=torch.float32))

        else:
            # === DETERMINISTYCZNE PATCHY DO WALIDACJI ===
            H, W, D = single_image_array_plus.shape
            out_x = min(128, H)
            out_y = min(128, W)
            out_z = min(16,  D)

            # --- 1. CENTROID FG (jeśli jest fg) ---
            fg_voxels = np.argwhere(single_image_array_plus > 0)

            if len(fg_voxels) > 0:
                cx, cy, cz = fg_voxels.mean(axis=0).astype(int)

                sx = np.clip(cx - out_x//2, 0, H - out_x)
                sy = np.clip(cy - out_y//2, 0, W - out_y)
                sz = np.clip(cz - out_z//2, 0, D - out_z)

                p_img = single_image_array[sx:sx+out_x, sy:sy+out_y, sz:sz+out_z]
                p_msk = single_image_array_plus[sx:sx+out_x, sy:sy+out_y, sz:sz+out_z]

                self.x.append(torch.tensor(p_img, dtype=torch.float32))
                self.y.append(torch.tensor(p_msk, dtype=torch.float32))

            # --- 2. CENTER CROP (zawsze, drugi patch) ---
            sx = (H - out_x)//2
            sy = (W - out_y)//2
            sz = (D - out_z)//2

            p_img = single_image_array[sx:sx+out_x, sy:sy+out_y, sz:sz+out_z]
            p_msk = single_image_array_plus[sx:sx+out_x, sy:sy+out_y, sz:sz+out_z]

            self.x.append(torch.tensor(p_img, dtype=torch.float32))
            self.y.append(torch.tensor(p_msk, dtype=torch.float32))

  def __getitem__(self, index):
    if torch.is_tensor(index):
            index = index.tolist()

    proccessed_out = {'image': self.x[index], 'mask': self.y[index]}

    return proccessed_out

  def __len__(self):
    return len(self.x)