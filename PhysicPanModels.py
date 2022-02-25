import torch
import torch.nn as nn

class MS2PAN(nn.Module):
    def __init__(self, spectrals):
        super(MS2PAN, self).__init__()
        self.in_dim = spectrals
        self.med_dim = spectrals * 2
        self.final_out_dim = 1

        self.layer0 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.med_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.med_dim, self.med_dim, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Conv2d(in_channels=self.med_dim, out_channels=self.final_out_dim, kernel_size=3, stride=1,
                                padding=1, bias=True)

    def forward(self, hms):
        y1 = self.layer0(hms)
        y2 = self.layer1(y1)

        res = torch.mean(hms, axis=1, keepdim=True) + self.layer2(y2)
        res.clamp(0.0, 1.0)
        return res


class HR2LRMS(nn.Module):
    def __init__(self, spectrals):
        super(HR2LRMS, self).__init__()
        self.layer_0 = nn.Sequential(
            nn.Conv2d(in_channels=spectrals, out_channels=spectrals*4, kernel_size=7, stride=1, padding=3, groups=spectrals),
            nn.Conv2d(in_channels=spectrals*4, out_channels=spectrals*4, kernel_size=5, stride=2, padding=2, groups=spectrals*2),
            nn.AvgPool2d(2),
            nn.ReLU()
        )

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=spectrals*4, out_channels=spectrals*2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=spectrals*2, out_channels=spectrals, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, hms):
        y1 = self.layer_0(hms)
        y2 = self.layer_1(y1)
        res = self.layer_2(y2)
        res.clamp(0.0, 1.0)
        return res

