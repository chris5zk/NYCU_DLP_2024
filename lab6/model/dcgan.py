# DCGAN
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, n_classes):
        super(Generator, self).__init__()
        self.nc = nc
        self.cond_emb = nn.Sequential(
            nn.Linear(n_classes, nc),
            nn.LeakyReLU(0.2, True)
        )
        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + nc, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, noise, cond):
        cond_emb = self.cond_emb(cond).view(-1, self.nc, 1, 1)
        net_in = torch.cat((noise, cond_emb), 1)
        out = self.net(net_in)
        return out


class Discriminator(nn.Module):
    def __init__(self, n_classes, ndf, img_size):
        super(Discriminator, self).__init__()
        self.cond_emb = nn.Sequential(
            nn.Linear(n_classes, img_size * img_size),
            nn.LeakyReLU(0.2, True)
        )
        self.net = nn.Sequential(
            # input is ``(3 + 1) x 64 x 64``
            nn.Conv2d(3 + 1, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, cond):
        cond_emb = self.cond_emb(cond).view(-1, 1, 64, 64)
        net_in = torch.cat((img, cond_emb), dim=1)
        out = self.net(net_in)
        return out.view(-1, 1).squeeze(1)