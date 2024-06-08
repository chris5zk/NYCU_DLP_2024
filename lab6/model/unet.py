# Unet
import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_residual=False):
        super().__init__()
        
        self.same_channels = (in_channels==out_channels) 
        self.is_residual = is_residual

        # conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_residual:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dims, n_classes=24):
        super(Down, self).__init__()
        self.dims = dims

        self.layer = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        )
        self.time_embed = Embed(1, dims)
        self.cond_embed = Embed(n_classes, dims)

    def forward(self, x, t, cond):
        t = self.time_embed(t).view(-1, self.dims, 1, 1)
        cond = self.cond_embed(cond).view(-1, self.dims, 1, 1)
        out  = self.layer(x * cond + t)
        return out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dims, n_classes=24):
        super(Up, self).__init__()
        self.dims = dims

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels)
        )
        self.time_embed = Embed(1, dims)
        self.cond_embed = Embed(n_classes, dims)

    def forward(self, x, skip, t, cond):
        t    = self.time_embed(t).view(-1, self.dims, 1, 1)
        cond = self.cond_embed(cond).view(-1, self.dims, 1, 1)
        out  = torch.cat((x * cond + t, skip), 1)
        out  = self.layer(out)
        return out

class Embed(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(Embed, self).__init__()

        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        out = x.view(-1, self.input_dim)
        out = self.model(out)
        return out

class Unet(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=256, n_classes=24):
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes

        """conv first"""
        self.initial_conv = ResidualConvBlock(in_channels, hidden_dims, is_residual=True)
        
        """down sampling"""
        self.down1 = Down(hidden_dims,     hidden_dims, dims=hidden_dims)
        self.down2 = Down(hidden_dims, 2 * hidden_dims, dims=hidden_dims)

        """bottom hidden of unet"""
        self.hidden = nn.Sequential(nn.AvgPool2d(8), nn.GELU())

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * hidden_dims, 2 * hidden_dims, 8, 8), 
            nn.GroupNorm(8, 2 * hidden_dims),
            nn.ReLU(),
        )
        self.up1 = Up(4 * hidden_dims, hidden_dims, dims=2*hidden_dims)
        self.up2 = Up(2 * hidden_dims, hidden_dims, dims=hidden_dims)

        """output"""
        self.out = nn.Sequential(
            nn.Conv2d(2 * hidden_dims, hidden_dims, 3, 1, 1),
            nn.GroupNorm(8, hidden_dims),
            nn.ReLU(),
            nn.Conv2d(hidden_dims, self.in_channels, 3, 1, 1),
        )

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x, cond, time):
        # initial conv
        x = self.initial_conv(x)  # [32,256,64,64]

        # down sampling
        down1 = self.down1(x, time, cond)
        down2 = self.down2(down1, time, cond)

        # hidden
        hidden = self.hidden(down2)

        # up sampling
        up1 = self.up0(hidden)                      # [32,256,64,64]
        up2 = self.up1(up1, down2, time, cond)  
        up3 = self.up2(up2, down1, time, cond)

        # output
        out = self.out(torch.cat((up3, x), 1))
        return out


if __name__ == '__main__':
    sample = torch.randn((32, 3, 64, 64)).to('cuda:1')
    cond = torch.ones((32, 24)).to('cuda:1')
    t = torch.ones(32).to('cuda:1')

    model = Unet().to('cuda:1')
    out = model(sample, cond, t)
    print(out.to('cpu').shape)