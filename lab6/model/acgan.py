# ACGAN
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
    def __init__(self, n_classes, ndf):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.net = nn.Sequential(
            # input is (rgb chnannel = 3) x 64 x 64
            nn.Conv2d(3, ndf, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*2) x 30 x 30
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size. (ndf*8) x 14 x 14
            nn.Conv2d(ndf * 8, ndf * 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # state size (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        
        # discriminator fc
        self.fc_dis = nn.Sequential(
            nn.Linear(5*5*ndf*32, 1),
            nn.Sigmoid()
        )
        # aux-classifier fc
        self.fc_aux = nn.Sequential(
            nn.Linear(5*5*ndf*32, n_classes),
            nn.Sigmoid()
        )
        
        
    def forward(self, input):
        conv = self.net(input)
        flat = conv.view(-1, 5*5*self.ndf*32)
        fc_dis = self.fc_dis(flat).view(-1, 1).squeeze(1)
        fc_aux = self.fc_aux(flat)
        return fc_dis, fc_aux