import torch
import torch.nn as nn

class MyUNet2D(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[32, 64, 128, 256]) -> None:
        super(MyUNet2D, self).__init__()

        #level 1
        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=level_channels[0]//2, kernel_size=3, padding=1)
        self.bn11 = nn.InstanceNorm2d(num_features=level_channels[0]//2)
        self.conv12 = nn.Conv2d(in_channels=level_channels[0]//2, out_channels=level_channels[0], kernel_size=3, padding=1)
        self.bn12 = nn.InstanceNorm2d(num_features=level_channels[0])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #level 2
        self.conv21 = nn.Conv2d(in_channels=level_channels[0], out_channels=level_channels[0], kernel_size=3, padding=1)
        self.bn21 = nn.InstanceNorm2d(num_features=level_channels[0])
        self.conv22 = nn.Conv2d(in_channels=level_channels[0], out_channels=level_channels[1], kernel_size=3, padding=1)
        self.bn22 = nn.InstanceNorm2d(num_features=level_channels[1])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #level 3
        self.conv31 = nn.Conv2d(in_channels=level_channels[1], out_channels=level_channels[1], kernel_size=3, padding=1)
        self.bn31 = nn.InstanceNorm2d(num_features=level_channels[1])
        self.conv32 = nn.Conv2d(in_channels=level_channels[1], out_channels=level_channels[2], kernel_size=3, padding=1)
        self.bn32 = nn.InstanceNorm2d(num_features=level_channels[2])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #level 4
        self.conv41 = nn.Conv2d(in_channels=level_channels[2], out_channels=level_channels[2], kernel_size=3, padding=1)
        self.bn41 = nn.InstanceNorm2d(num_features=level_channels[2])
        self.conv42 = nn.Conv2d(in_channels=level_channels[2], out_channels=level_channels[3], kernel_size=3, padding=1)
        self.bn42 = nn.InstanceNorm2d(num_features=level_channels[3])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #bottleneck
        self.conv51 = nn.Conv2d(in_channels=level_channels[3], out_channels=level_channels[3], kernel_size=3, padding=1)
        self.bn51 = nn.InstanceNorm2d(num_features=level_channels[3])
        self.conv52 = nn.Conv2d(in_channels=level_channels[3], out_channels=level_channels[3]*2, kernel_size=3, padding=1)
        self.bn52 = nn.InstanceNorm2d(num_features=level_channels[3]*2)
        self.relu = nn.ReLU()
        self.upconv5 = nn.ConvTranspose2d(in_channels=level_channels[3]*2, out_channels=level_channels[3], kernel_size=2, stride=2)

        #level 1 up
        self.conv61 = nn.Conv2d(in_channels=level_channels[3]*2, out_channels=level_channels[3], kernel_size=3, padding=1)
        self.bn61 = nn.InstanceNorm2d(num_features=level_channels[3])
        self.conv62 = nn.Conv2d(in_channels=level_channels[3], out_channels=level_channels[3], kernel_size=3, padding=1)
        self.bn62 = nn.InstanceNorm2d(num_features=level_channels[3])
        self.relu = nn.ReLU()
        self.upconv6 = nn.ConvTranspose2d(in_channels=level_channels[3], out_channels=level_channels[2], kernel_size=2, stride=2)

        #level 2 up
        self.conv71 = nn.Conv2d(in_channels=level_channels[2]*2, out_channels=level_channels[2], kernel_size=3, padding=1)
        self.bn71 = nn.InstanceNorm2d(num_features=level_channels[2])
        self.conv72 = nn.Conv2d(in_channels=level_channels[2], out_channels=level_channels[2], kernel_size=3, padding=1)
        self.bn72 = nn.InstanceNorm2d(num_features=level_channels[2])
        self.relu = nn.ReLU()
        self.upconv7 = nn.ConvTranspose2d(in_channels=level_channels[2], out_channels=level_channels[1], kernel_size=2, stride=2)

        #level 3 up
        self.conv81 = nn.Conv2d(in_channels=level_channels[1]*2, out_channels=level_channels[1], kernel_size=3, padding=1)
        self.bn81 = nn.InstanceNorm2d(num_features=level_channels[1])
        self.conv82 = nn.Conv2d(in_channels=level_channels[1], out_channels=level_channels[1], kernel_size=3, padding=1)
        self.bn82 = nn.InstanceNorm2d(num_features=level_channels[1])
        self.relu = nn.ReLU()
        self.upconv8 = nn.ConvTranspose2d(in_channels=level_channels[1], out_channels=level_channels[0], kernel_size=2, stride=2)

        #level 4 up
        self.conv91 = nn.Conv2d(in_channels=level_channels[0]*2, out_channels=level_channels[0], kernel_size=3, padding=1)
        self.bn91 = nn.InstanceNorm2d(num_features=level_channels[0])
        self.conv92 = nn.Conv2d(in_channels=level_channels[0], out_channels=level_channels[0], kernel_size=3, padding=1)
        self.bn92 = nn.InstanceNorm2d(num_features=level_channels[0])
        self.relu = nn.ReLU()

        #last
        self.conv9 = nn.Conv2d(in_channels=level_channels[0], out_channels=num_classes, kernel_size=1)

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

        #level 4
        res4 = self.relu(self.bn41(self.conv41(out3)))
        res4 = self.relu(self.bn42(self.conv42(res4)))
        out4 = self.pool(res4)

        #bottleneck
        res_btl = self.relu(self.bn51(self.conv51(out4)))
        res_btl = self.relu(self.bn52(self.conv52(res_btl)))
        out_btl = self.upconv5(res_btl)

        #level 1 up
        out4 = torch.cat((res4, out_btl), 1)
        out4 = self.relu(self.bn61(self.conv61(out4)))
        out4 = self.relu(self.bn62(self.conv62(out4)))
        out4 = self.upconv6(out4)

        #level 2 up
        out5 = torch.cat((res3, out4), 1)
        out5 = self.relu(self.bn71(self.conv71(out5)))
        out5 = self.relu(self.bn72(self.conv72(out5)))
        out5 = self.upconv7(out5)

        #level 3 up
        out6 = torch.cat((res2, out5), 1)
        out6 = self.relu(self.bn81(self.conv81(out6)))
        out6 = self.relu(self.bn82(self.conv82(out6)))
        out6 = self.upconv8(out6)

        #level 4 up
        out7 = torch.cat((res1, out6), 1)
        out7 = self.relu(self.bn91(self.conv91(out7)))
        out7 = self.relu(self.bn92(self.conv92(out7)))

        #last
        out8 = self.conv9(out7)

        return out8


class MyUNet3D(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[32, 64, 128], bottleneck_channel=256) -> None:
        super(MyUNet3D, self).__init__()

        #level 1
        self.conv11 = nn.Conv3d(in_channels= in_channels, out_channels=level_channels[0]//2, kernel_size=(3,3,3), padding=1)
        self.bn11 = nn.BatchNorm3d(num_features=level_channels[0]//2)
        self.conv12 = nn.Conv3d(in_channels= level_channels[0]//2, out_channels=level_channels[0], kernel_size=(3,3,3), padding=1)
        self.bn12 = nn.BatchNorm3d(num_features=level_channels[0])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

        #level 2
        self.conv21 = nn.Conv3d(in_channels= level_channels[0], out_channels=level_channels[0], kernel_size=(3,3,3), padding=1)
        self.bn21 = nn.BatchNorm3d(num_features=level_channels[0])
        self.conv22 = nn.Conv3d(in_channels= level_channels[0], out_channels=level_channels[1], kernel_size=(3,3,3), padding=1)
        self.bn22 = nn.BatchNorm3d(num_features=level_channels[1])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

        #level 3
        self.conv31 = nn.Conv3d(in_channels= level_channels[1], out_channels=level_channels[1], kernel_size=(3,3,3), padding=1)
        self.bn31 = nn.BatchNorm3d(num_features=level_channels[1])
        self.conv32 = nn.Conv3d(in_channels= level_channels[1], out_channels=level_channels[2], kernel_size=(3,3,3), padding=1)
        self.bn32 = nn.BatchNorm3d(num_features=level_channels[2])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

        #bottleneck
        self.conv41 = nn.Conv3d(in_channels= level_channels[2], out_channels=level_channels[2], kernel_size=(3,3,3), padding=1)
        self.bn41 = nn.BatchNorm3d(num_features=level_channels[2])
        self.conv42 = nn.Conv3d(in_channels= level_channels[2], out_channels=level_channels[2]*2, kernel_size=(3,3,3), padding=1)
        self.bn42 = nn.BatchNorm3d(num_features=level_channels[2]*2)
        self.relu = nn.ReLU()
        self.upconv4 = nn.ConvTranspose3d(in_channels=level_channels[2]*2, out_channels=level_channels[2]*2, kernel_size=(2, 2, 2), stride=2)

        #level 1 up
        self.conv51 = nn.Conv3d(in_channels= level_channels[2]*2, out_channels=level_channels[2], kernel_size=(3,3,3), padding=1)
        self.bn51 = nn.BatchNorm3d(num_features=level_channels[2])
        self.conv52 = nn.Conv3d(in_channels= level_channels[2], out_channels=level_channels[2], kernel_size=(3,3,3), padding=1)
        self.bn52 = nn.BatchNorm3d(num_features=level_channels[2])
        self.relu = nn.ReLU()
        self.upconv5 = nn.ConvTranspose3d(in_channels=level_channels[2], out_channels=level_channels[2], kernel_size=(2, 2, 2), stride=2)

        #level 2 up
        self.conv61 = nn.Conv3d(in_channels= level_channels[1]*2, out_channels=level_channels[1], kernel_size=(3,3,3), padding=1)
        self.bn61 = nn.BatchNorm3d(num_features=level_channels[1])
        self.conv62 = nn.Conv3d(in_channels= level_channels[1], out_channels=level_channels[1], kernel_size=(3,3,3), padding=1)
        self.bn62 = nn.BatchNorm3d(num_features=level_channels[1])
        self.relu = nn.ReLU()
        self.upconv6 = nn.ConvTranspose3d(in_channels=level_channels[1], out_channels=level_channels[1], kernel_size=(2, 2, 2), stride=2)

        #level 3 up
        self.conv71 = nn.Conv3d(in_channels= level_channels[0]*2, out_channels=level_channels[0], kernel_size=(3,3,3), padding=1)
        self.bn71 = nn.BatchNorm3d(num_features=level_channels[0])
        self.conv72 = nn.Conv3d(in_channels= level_channels[0], out_channels=level_channels[0], kernel_size=(3,3,3), padding=1)
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
        print(res3.shape)
        print(out_btl.shape)
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