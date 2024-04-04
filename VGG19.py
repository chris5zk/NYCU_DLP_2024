import torch.nn as nn


class vgg19(nn.Module):
    def __init__(self, num_classes=1000) -> None:
        super(vgg19, self).__init__()
        
        # backbone
        self.block1 = self.build_layer(in_channels=3, out_channels=64, conv_nums=2)
        self.block2 = self.build_layer(in_channels=64, out_channels=128, conv_nums=2)
        self.block3 = self.build_layer(in_channels=128, out_channels=256, conv_nums=4)
        self.block4 = self.build_layer(in_channels=256, out_channels=512, conv_nums=4)
        self.block5 = self.build_layer(in_channels=512, out_channels=512, conv_nums=4)
        
        # prediction head
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
    def build_layer(self, in_channels, out_channels, conv_nums):
        layers = []
        
        # first conv. layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # rest conv. layer
        for _ in range(conv_nums-1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # max-pooling layer
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    # load model
    model = vgg19()
    print(model)

    # pseudo input
    import torch
    x = torch.randn(1, 3, 224, 224)
    y_pred = model(x)
    
    print(y_pred)