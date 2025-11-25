import nibabel as nib
import glob
import numpy as np
import matplotlib.pyplot as plt

patients = glob.glob("NII/*")
path = patients[0]

single_img = nib.load(path + "/segmentations/Hippocampus_L.nii.gz").get_fdata()

height, width, depth = single_img.shape
print(f"The image object has the following dimensions: height: {height}, width:{width}, depth:{depth}")

i = 55
print(f"Plotting Layer {i} of Image")
plt.imshow(single_img[:, :, i], cmap="gray")
plt.show()
print(f'With the unique values: {np.unique(single_img)}')

single_img[single_img == 255] = 1
img = single_img[:,:,i]
mask = np.where(img == 1, 255, 0)
plt.imshow(mask, cmap="gray")
plt.show()