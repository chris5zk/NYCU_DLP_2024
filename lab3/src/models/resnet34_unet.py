import torch
import torch.nn as nn
import torch.nn.functional as F

class resnet34_unet(nn.Module):
    def __init__(self) -> None:
        super(resnet34_unet, self).__init__()
        
        # encoder
        self.enc_conv1 = self._double_conv(3, 64)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_conv2 = self._double_conv(64, 128)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_conv3 = self._double_conv(128, 256)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.enc_conv4 = self._double_conv(256, 512)
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # bottle_neck
        self.bottle_neck = self._double_conv(512, 1024)
        
        # decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.dec_conv1 = self._double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.dec_conv2 = self._double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.dec_conv3 = self._double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.dec_conv4 = self._double_conv(128, 64)

        # output
        self.output = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def _double_conv(self, in_channels, out_channels):
            return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
    
    def basic_block(self, in_channels, out_channels):
        return
    
    def forward(self, x):
        x1 = self.enc_conv1(x)
        x = self.down1(x1)
        
        x2 = self.enc_conv2(x)
        x = self.down2(x2)
        
        x3 = self.enc_conv3(x)
        x = self.down3(x3)
        
        x4 = self.enc_conv4(x)
        x = self.down1(x4)
        
        x = self.bottle_neck(x)
        
        x = self.up1(x)
        x = self.dec_conv1(torch.cat([x4, x], dim=1))
        
        x = self.up2(x)
        x = self.dec_conv2(torch.cat([x3, x], dim=1))

        x = self.up3(x)
        x = self.dec_conv3(torch.cat([x2, x], dim=1))

        x = self.up4(x)
        x = self.dec_conv4(torch.cat([x1, x], dim=1))

        x = self.output(x)
        x = self.sigmoid(x)

        return x



    
if __name__ == '__main__':
    model = resnet34_unet()
    print(model)
    
    # pseudo input
    x = torch.randn(1, 3, 256, 256)
    y_pred = model(x)
    
    print(x.shape, y_pred.shape)