import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import math
import nibabel as nib
from torch.utils.tensorboard import SummaryWriter

def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3),
                   epsilon=0.00001):

    dice_numerator = 2 * torch.sum(y_true*y_pred) + epsilon
    dice_denominator = torch.sum(y_pred*y_pred) + torch.sum(y_true*y_true) + epsilon
    dice_loss = 1 - torch.mean(dice_numerator / dice_denominator)

    return dice_loss

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

class Unet3D(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Unet3D, self).__init__(*args, **kwargs)

        #down
        self.down_depth_0_layer_0_conv = nn.Conv3d(in_channels=1, out_channels=8,
                              kernel_size=(3,3,3),
                              padding='same',
                              stride=(1,1,1)
                              )
        self.down_depth_0_layer_0_relu = nn.ReLU()

        self.down_depth_0_layer_1_conv = nn.Conv3d(in_channels=8, out_channels=16,
                kernel_size=(3,3,3),
                padding='same',
                stride=(1,1,1)
               )
        self.down_depth_0_layer_1_relu = nn.ReLU()

        self.down_depth_0_layer_pool = nn.MaxPool3d(kernel_size=(2,2,2))

        self.down_depth_1_layer_0_conv = nn.Conv3d(in_channels=16, out_channels=16,
                kernel_size=(3,3,3),
                padding='same',
                stride=(1,1,1)
               )
        self.down_depth_1_layer_0_relu = nn.ReLU()

        self.down_depth_1_layer_1_conv = nn.Conv3d(in_channels=16, out_channels=32,
                kernel_size=(3,3,3),
                padding='same',
                stride=(1,1,1)
               )
        self.down_depth_1_layer_1_relu = nn.ReLU()

        #up
        self.up_depth_1_layer_1_conv = nn.Conv3d(in_channels=48, out_channels=16,
                kernel_size=(3,3,3),
                padding='same',
                stride=(1,1,1)
               )
        self.up_depth_1_layer_1_relu = nn.ReLU()

        self.up_depth_1_layer_2_conv = nn.Conv3d(in_channels=16, out_channels=8,
                kernel_size=(3,3,3),
                padding='same',
                stride=(1,1,1)
               )
        self.up_depth_1_layer_2_relu = nn.ReLU()

        self.final_conv = nn.Conv3d(in_channels=8, out_channels=1,
                kernel_size=(1,1,1),
                padding='valid',
                stride=(1,1,1)
               )
        self.final_sigma = nn.Sigmoid()

    def forward(self, input_layer):

        down_depth_0_layer_0 = self.down_depth_0_layer_0_conv(input_layer)
        down_depth_0_layer_0 = self.down_depth_0_layer_0_relu(down_depth_0_layer_0)

        down_depth_0_layer_1 = self.down_depth_0_layer_1_conv(down_depth_0_layer_0)
        down_depth_0_layer_1 = self.down_depth_0_layer_1_relu(down_depth_0_layer_1)
        down_depth_0_layer_pooled = self.down_depth_0_layer_pool(down_depth_0_layer_1)

        down_depth_1_layer_0 = self.down_depth_1_layer_0_conv(down_depth_0_layer_pooled)
        down_depth_1_layer_0 = self.down_depth_1_layer_0_relu(down_depth_1_layer_0)

        down_depth_1_layer_1 = self.down_depth_1_layer_1_conv(down_depth_1_layer_0)
        down_depth_1_layer_1 = self.down_depth_1_layer_1_relu(down_depth_1_layer_1)

        down_depth_1_layer_1 = down_depth_1_layer_1.unsqueeze(0)
        up_depth_0_layer_0 = nn.functional.interpolate(down_depth_1_layer_1, scale_factor=(2,2,2))

        down_depth_0_layer_1 = down_depth_0_layer_1.unsqueeze(0)
        up_depth_1_concat = torch.cat((up_depth_0_layer_0,
                                        down_depth_0_layer_1),
                                        dim=1)

        up_depth_1_layer_1 = self.up_depth_1_layer_1_conv(up_depth_1_concat)
        up_depth_1_layer_1 = self.up_depth_1_layer_1_relu(up_depth_1_layer_1)

        up_depth_1_layer_2 = self.up_depth_1_layer_2_conv(up_depth_1_layer_1)
        up_depth_1_layer_2 = self.up_depth_1_layer_2_relu(up_depth_1_layer_2)

        final = self.final_conv(up_depth_1_layer_2)
        final = self.final_sigma(final)

        return final
    
class MyUNet3D(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[8, 16, 32], bottleneck_channel=64) -> None:
        super(MyUNet3D, self).__init__()

        #level 1
        self.conv11 = nn.Conv3d(in_channels=in_channels, out_channels=level_channels[0]//2, kernel_size=(3,3,3), padding=1)
        self.bn11 = nn.BatchNorm3d(num_features=level_channels[0]//2)
        self.conv12 = nn.Conv3d(in_channels=level_channels[0]//2, out_channels=level_channels[0], kernel_size=(3,3,3), padding=1)
        self.bn12 = nn.BatchNorm3d(num_features=level_channels[0])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

        #level 2
        self.conv21 = nn.Conv3d(in_channels=level_channels[0], out_channels=level_channels[0], kernel_size=(3,3,3), padding=1)
        self.bn21 = nn.BatchNorm3d(num_features=level_channels[0])
        self.conv22 = nn.Conv3d(in_channels=level_channels[0], out_channels=level_channels[1], kernel_size=(3,3,3), padding=1)
        self.bn22 = nn.BatchNorm3d(num_features=level_channels[1])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

        #level 3
        self.conv31 = nn.Conv3d(in_channels=level_channels[1], out_channels=level_channels[1], kernel_size=(3,3,3), padding=1)
        self.bn31 = nn.BatchNorm3d(num_features=level_channels[1])
        self.conv32 = nn.Conv3d(in_channels=level_channels[1], out_channels=level_channels[2], kernel_size=(3,3,3), padding=1)
        self.bn32 = nn.BatchNorm3d(num_features=level_channels[2])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

        #bottleneck
        self.conv41 = nn.Conv3d(in_channels=level_channels[2], out_channels=level_channels[2], kernel_size=(3,3,3), padding=1)
        self.bn41 = nn.BatchNorm3d(num_features=level_channels[2])
        self.conv42 = nn.Conv3d(in_channels=level_channels[2], out_channels=level_channels[2]*2, kernel_size=(3,3,3), padding=1)
        self.bn42 = nn.BatchNorm3d(num_features=level_channels[2]*2)
        self.relu = nn.ReLU()
        self.upconv4 = nn.ConvTranspose3d(in_channels=level_channels[2]*2, out_channels=level_channels[2]*2, kernel_size=(2, 2, 2), stride=2)

        #level 1 up
        self.conv51 = nn.Conv3d(in_channels=level_channels[2]*3, out_channels=level_channels[2], kernel_size=(3,3,3), padding=1)
        self.bn51 = nn.BatchNorm3d(num_features=level_channels[2])
        self.conv52 = nn.Conv3d(in_channels=level_channels[2], out_channels=level_channels[2], kernel_size=(3,3,3), padding=1)
        self.bn52 = nn.BatchNorm3d(num_features=level_channels[2])
        self.relu = nn.ReLU()
        self.upconv5 = nn.ConvTranspose3d(in_channels=level_channels[2], out_channels=level_channels[2], kernel_size=(2, 2, 2), stride=2)

        #level 2 up
        self.conv61 = nn.Conv3d(in_channels=level_channels[1]*3, out_channels=level_channels[1], kernel_size=(3,3,3), padding=1)
        self.bn61 = nn.BatchNorm3d(num_features=level_channels[1])
        self.conv62 = nn.Conv3d(in_channels=level_channels[1], out_channels=level_channels[1], kernel_size=(3,3,3), padding=1)
        self.bn62 = nn.BatchNorm3d(num_features=level_channels[1])
        self.relu = nn.ReLU()
        self.upconv6 = nn.ConvTranspose3d(in_channels=level_channels[1], out_channels=level_channels[1], kernel_size=(2, 2, 2), stride=2)

        #level 3 up
        self.conv71 = nn.Conv3d(in_channels=level_channels[0]*3, out_channels=level_channels[0], kernel_size=(3,3,3), padding=1)
        self.bn71 = nn.BatchNorm3d(num_features=level_channels[0])
        self.conv72 = nn.Conv3d(in_channels=level_channels[0], out_channels=level_channels[0], kernel_size=(3,3,3), padding=1)
        self.bn72 = nn.BatchNorm3d(num_features=level_channels[0])
        self.relu = nn.ReLU()

        #last
        self.conv8 = nn.Conv3d(in_channels=level_channels[0], out_channels=num_classes, kernel_size=(1,1,1))

    def forward(self, input):
        input = input.float()
        #level 1
        res1 = self.relu(self.bn11(self.conv11(input)))
        res1 = self.relu(self.bn12(self.conv12(res1)))
        out1 = self.pool(res1)

        #level 2
        res2 = self.relu(self.bn21(self.conv21(out1)))
        res2 = self.relu(self.bn22(self.conv22(res2)))
        out2 = self.pool(res2)

        #level 3
        res3 = self.relu(self.bn31(self.conv31(out2)))
        res3 = self.relu(self.bn32(self.conv32(res3)))
        out3 = self.pool(res3)

        #bottleneck
        res_btl = self.relu(self.bn41(self.conv41(out3)))
        res_btl = self.relu(self.bn42(self.conv42(res_btl)))
        out_btl = self.upconv4(res_btl)

        #level 1 up
        out4 = torch.cat((res3, out_btl), 1)
        out4 = self.relu(self.bn51(self.conv51(out4)))
        out4 = self.relu(self.bn52(self.conv52(out4)))
        out4 = self.upconv5(out4)

        #level 2 up
        out5 = torch.cat((res2, out4), 1)
        out5 = self.relu(self.bn61(self.conv61(out5)))
        out5 = self.relu(self.bn62(self.conv62(out5)))
        out5 = self.upconv6(out5)

        #level 3 up
        out6 = torch.cat((res1, out5), 1)
        out6 = self.relu(self.bn71(self.conv71(out6)))
        out6 = self.relu(self.bn72(self.conv72(out6)))

        #last
        out7 = self.conv8(out6)

        return out7
    
test_data = DatasetFromNii(image_dir='/content/train_set/')
val_data = DatasetFromNii(image_dir='/content/val_set/')

train_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
val_dataloader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)
    

model = MyUNet3D(in_channels=1, num_classes=1)
criterion = BCEWithLogitsLoss()
optimizer = Adam(params=model.parameters())

writer = SummaryWriter("runs")

min_valid_loss = math.inf

for epoch in range(40):

    train_loss = 0.0
    model.train()
    for data in train_dataloader:
        image, ground_truth = data['image'], data['mask']
        image = image[:, None, :, :, :]
        ground_truth = ground_truth[:, None, :, :, :]
        optimizer.zero_grad()
        target = model(image)

        #loss = criterion(target, ground_truth)
        loss = soft_dice_loss(F.sigmoid(target), ground_truth)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()
    for data in val_dataloader:
        image, ground_truth = data['image'], data['mask']
        image = image[:, None, :, :, :]
        ground_truth = ground_truth[:, None, :, :, :]
        target = model(image)

        #loss = criterion(target, ground_truth)
        loss = soft_dice_loss(F.sigmoid(target), ground_truth)
        valid_loss += loss.item()

    frame = 8
    figure, axis = plt.subplots(1, 2)
    target = np.where(F.sigmoid(target) > 0.5, 255, 0)
    axis[0].imshow(target[0,0,:,:,frame], cmap="gray")
    axis[1].imshow(ground_truth[0,0,:,:,frame], cmap="gray")
    plt.show()

    writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
    writer.add_scalar("Loss/Validation", valid_loss / len(val_dataloader), epoch)

    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({(min_valid_loss / len(val_dataloader)):.6f}--->{(valid_loss / len(val_dataloader)):.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), f'checkpoints/epoch{epoch}_valLoss{(min_valid_loss / len(val_dataloader))}.pth')

writer.flush()
writer.close()