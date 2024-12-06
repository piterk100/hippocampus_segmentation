import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import math
from torch.utils.tensorboard import SummaryWriter
import unet
import dataloader
    
test_data = dataloader.DatasetFromNii(image_dir='/content/train_set/')
val_data = dataloader.DatasetFromNii(image_dir='/content/val_set/')

train_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
val_dataloader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)
    

model = unet.MyUNet3D(in_channels=1, num_classes=1)
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
        loss = unet.soft_dice_loss(F.sigmoid(target), ground_truth)
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
        loss = unet.soft_dice_loss(F.sigmoid(target), ground_truth)
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