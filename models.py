
import torch
import torch.nn as nn

from util import *

def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x/10*1.28

    variance_scaling(tensor)

    return tensor
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):   ## initialization for Conv2d
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)  # method 1: initialization
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):   ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):     ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class Residual_BlockWOBN(nn.Module):
    def __init__(self, channels, ker_size=3, padd=1):
        super(Residual_BlockWOBN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=ker_size, stride=1, padding=padd)
        self.lrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=ker_size, stride=1, padding=padd)

    def forward(self, x):
        fea = self.lrelu((self.conv1(x)))
        fea = self.lrelu((self.conv2(fea)))

        result = fea + x
        return result



class PyramidInjection(nn.Module):
    def __init__(self, spectrals, downscales):
        super(PyramidInjection, self).__init__()
        self.spectrals = spectrals
        self.DenseLayers = 3
        self.midChannels = 32 #spectrals * 4
        self.downscales = downscales

        self.blocks1 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=spectrals * 2, out_channels=self.midChannels, kernel_size=5, stride=1, padding=2, groups=spectrals),
            nn.ReLU(True),
            nn.Conv2d(in_channels=self.midChannels, out_channels=self.midChannels, kernel_size=3, stride=2, padding=1)
        ) for i in range(self.downscales)])

        self.blocks2 = nn.ModuleList([nn.Sequential(
            self.make_ResLayer(Residual_BlockWOBN, 2, self.midChannels),
        ) for i in range(self.downscales)])

        self.blocks3 = nn.ModuleList([self.make_DenseLayer(self.DenseLayers) for i in range(self.downscales)])

        self.blocks4 = nn.ModuleList([nn.Sequential(
            nn.Conv2d((self.DenseLayers + 1) * self.midChannels, self.midChannels, 1, padding=0, stride=1),
            self.make_ResLayer(Residual_BlockWOBN, 2, self.midChannels),
        ) for i in range(self.downscales)])

        self.laterals = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=self.midChannels, out_channels=self.midChannels, kernel_size=3, stride=1, padding=1)
        ) for i in range(self.downscales)])

        self.blocks5 = nn.Sequential(
            nn.Conv2d(in_channels=self.midChannels, out_channels=spectrals * 2, kernel_size=3, stride=1, padding=1,
                               bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=spectrals * 2, out_channels=spectrals, kernel_size=3, stride=1, padding=1,
                      bias=True)
        )
        # init_weights(self.blocks1, self.blocks2, self.blocks3, self.blocks4, self.blocks5)

    def make_ResLayer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)
    def make_DenseLayer(self, num_layers):
        layer_list = []
        for i in range(num_layers):
            layer_list.append(nn.Sequential(
                nn.Conv2d(in_channels=self.midChannels + i * self.midChannels, out_channels=self.midChannels, kernel_size=3, stride=1, padding=1,
                          bias=True),
                nn.ReLU(True),
            ))
        return nn.Sequential(*layer_list)

    def forward(self, mss, pan):
        lr_u_hp = mss

        input_bundle = [[] for i in range(self.downscales)]

        pan_down_highs = [pan]
        mss_down_highs = [mss]
        for i in range(1, self.downscales):
            pan_down = nn.functional.interpolate(pan, scale_factor=1. / (2 ** i), mode='bilinear', align_corners=False,
                                               recompute_scale_factor=True)
            pan_down_highs.append(pan_down)
            mss_down = nn.functional.interpolate(mss, scale_factor=1. / (2 ** i), mode='bilinear', align_corners=False,
                                               recompute_scale_factor=True)
            mss_down_highs.append(mss_down)

        for i in range(self.downscales):
            mss_high = mss_down_highs[i]
            pan_high = pan_down_highs[i]
            for channIndex in range(self.spectrals):
                cur_cat = torch.cat((mss_high[:, channIndex, ].unsqueeze(1), pan_high), 1)
                input_bundle[i].append(cur_cat)
        for i in range(self.downscales):
            input_bundle[i] = torch.cat(input_bundle[i], 1)

        feat1 = [self.blocks1[i](input_bundle[i]) for i in range(self.downscales)]

        feat2 = [self.blocks2[i](feat1[i]) for i in range(self.downscales)]

        feat3 = [0]*self.downscales
        for idx in range(self.downscales):
            feature = self.blocks3[idx][0](feat2[idx])
            new_feat = torch.cat((feat2[idx], feature), 1)
            cur_feat = [new_feat]
            for i in range(1, self.DenseLayers):
                new_feat = self.blocks3[idx][i](new_feat)
                cur_feat.append(new_feat)
                new_feat = torch.cat(cur_feat, 1)
            feat3[idx] = torch.cat(cur_feat, 1)

        feat4 = [self.blocks4[i](feat3[i]) for i in range(self.downscales)]
        # med_map = feat4[0][0, 1, :, :] * 255
        # med_map = Image.fromarray(med_map.cpu().detach().numpy().astype(np.uint8))
        # med_map.save("1.jpg")
        coarse = nn.functional.interpolate(self.laterals[-1](feat4[-1]), scale_factor=2, mode="nearest")
        for i in range(1, self.downscales):
            coarse = nn.functional.interpolate(self.laterals[-i-1](feat4[-i-1]) + coarse, scale_factor=2, mode="nearest")

        fea_out = coarse

        res = torch.tanh(self.blocks5(fea_out)) + mss

        return torch.clamp(res, min=0, max=1)
