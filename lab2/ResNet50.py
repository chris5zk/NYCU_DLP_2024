import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, channels, stride=1, expansion=4, downsampling=False):
        super(Bottleneck,self).__init__()
        self.downsampling = downsampling
        
        self.expansion = expansion
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=channels,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels*self.expansion),
        )
        
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels*self.expansion)
            )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        
        if self.downsampling:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out


class resnet50(nn.Module):
    def __init__(self, num_classes=100, expansion = 4):
        super(resnet50, self).__init__()
        self.expansion = expansion
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.stage2 = self.build_layer(  64,  64, 3, 1) # 64 64 256
        self.stage3 = self.build_layer( 256, 128, 4, 2) # 128 128 512
        self.stage4 = self.build_layer( 512, 256, 6, 2) # 256 256 1024
        self.stage5 = self.build_layer(1024, 512, 3, 2) # 512 512 2048
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)
        
    def build_layer(self, in_channels, channels, conv_nums, stride):
        layers = []
        layers.append(Bottleneck(in_channels, channels, stride, downsampling=True))
        for _ in range(1, conv_nums):
            layers.append(Bottleneck(channels * self.expansion, channels))
        return nn.Sequential(*layers)
        
    
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    model = resnet50()
    print(model)
    
    # pseudo input
    import torch
    x = torch.randn(4, 3, 224, 224)
    y_pred = model(x)
    
    print(y_pred.shape)
    