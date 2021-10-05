# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from collections.abc import Iterable
import torch.nn.functional as F
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import LinearizedConvolution
from examples.audio_visual_se_sr.data.data_utils import lengths_to_encoder_padding_mask
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer, VGGBlock, TransformerBiModalDecoderLayer, MultiheadAttention
from fairseq.modules import LayerNorm
import pdb


def calc_mean_invstddev(feature):
    if len(feature.size()) != 2:
        raise ValueError("We expect the input feature to be 2-D tensor")
    mean = feature.mean(0)
    var = feature.var(0)
    # avoid division by ~zero
    eps = 1e-8
    if (var < eps).any():
        return mean, 1.0 / (torch.sqrt(var) + eps)
    return mean, 1.0 / torch.sqrt(var)


def apply_mv_norm(features):
    
    for i in range(features.size(0)):
        if i ==0:
            mean, invstddev = calc_mean_invstddev(features[i])
            res = (features[i] - mean) * invstddev
            res = res.unsqueeze(0)
        else:
            mean, invstddev = calc_mean_invstddev(features[i])
            res1 = (features[i] - mean) * invstddev
            res1 = res1.unsqueeze(0)
            res = torch.cat([res,res1],dim=0)
    return res

class ResidualBlock_transcnn(nn.Module):
    def __init__(self,in_channel, out_channel,stride):
        super(ResidualBlock_transcnn,self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=stride,padding=2,groups=in_channel),
                nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                nn.BatchNorm1d(out_channel),
                nn.ReLU(),

                )
        self.layer2 = nn.Sequential(
                nn.ConvTranspose1d(in_channel, out_channel, kernel_size=5,stride=stride,padding=2,output_padding=1,groups=out_channel),
                nn.Conv1d(out_channel, out_channel,kernel_size=1,stride=1,padding=0),
                nn.BatchNorm1d(out_channel),
                nn.ReLU(),
                )
        self.upsample = nn.Upsample(scale_factor=2,mode='nearest')
        self.stride=stride
    def forward(self,x):
        residual = x
        
        if self.stride==1:
          out = self.layer1(x)

          if residual.shape[2] < out.shape[2]:
              residual = self.upsample(residual)
          out += residual
        
        if self.stride==2:
          out = self.layer2(x)
          if residual.shape[2] < out.shape[2]:  ## time compare
              residual = self.upsample(residual)

          out += residual
        
        return out


class Resnet_transcnn(nn.Module):
    def __init__(self, block, init_channel, out_channel,blockslayers,stride):
        super(Resnet_transcnn, self).__init__()
        self.layer=self.make_layer(block, init_channel,out_channel, blockslayers,stride)


    def make_layer(self, block, init_channel,out_channel, blockslayers, stride):
        layers=[]
        layers.append(block(init_channel, out_channel[0],stride[0]))
        for i in range(blockslayers-1):
            layers.append(block(out_channel[i],out_channel[i+1],stride[i+1]))
        #print(layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        out=self.layer(x)
        return out

class ResidualBlock_cnn(nn.Module):
    def __init__(self,in_channel, out_channel,stride):
        super(ResidualBlock_cnn,self).__init__()
        self.layer = nn.Sequential(
                nn.Conv1d(in_channel, in_channel, kernel_size=5,stride=stride,padding=2,groups=in_channel),
                nn.Conv1d(in_channel, out_channel,kernel_size=1,stride=1,padding=0),
                nn.BatchNorm1d(out_channel),
                nn.ReLU(),
                )
        self.upchannel = nn.Conv1d(in_channel, out_channel,  kernel_size=1,stride=1)
        
        
        
    def forward(self,x):
        residual = x
        #print(x.shape)
        out = self.layer(x)
        #print(out.shape)
        if residual.shape[1] < out.shape[1]: # channel compare
            residual = self.upchannel(residual)
            #print(residual.shape)

        
        out += residual
        return out


class Resnet_cnn(nn.Module):
    def __init__(self, block, init_channel,out_channel,blockslayers,stride):
        super(Resnet_cnn, self).__init__()
        self.layer=self.make_layer(block,init_channel, out_channel, blockslayers,stride)

    def make_layer(self, block, init_channel,out_channel, blockslayers, stride):
        layers=[]
        layers.append(block(init_channel, out_channel[0],stride[0]))
        for i in range(blockslayers-1):
            layers.append(block(out_channel[i],out_channel[i+1],stride[i+1]))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out=self.layer(x)
        return out


class ComplexConv2d(nn.Module):
    # https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output




class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, complex=False, padding_mode="zeros"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding
            
        if complex:
            conv = ComplexConv2d
            bn = ComplexBatchNorm2d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding,padding=(0, 0), complex=False):
        super().__init__()
        if complex:
            tconv = ComplexConvTranspose2d
            bn = ComplexBatchNorm2d
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
        
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=output_padding,padding=padding)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResPath(nn.Module):
    def __init__(self, channel):
        super().__init__()
    
        self.conv1 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.resconv1 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.resconv2 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.resconv3 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
        self.resconv4 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1,stride=1,padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                )
    def forward(self, x):
       
        x1 = self.conv1(x)
        y1 = self.resconv1(x)
        z1 = x1+y1
        
        x2 = self.conv2(z1)
        y2 = self.resconv2(z1)
        z2 = x2+y2
        
        x3 = self.conv3(z2)
        y3 = self.resconv3(z2)
        z3 = x3+y3
        
        x4 = self.conv4(z3)
        y4 = self.resconv4(z3)
        z4 = x4+y4
        
        return z4

class UNet(nn.Module):
    def __init__(self,input_channels=2,complex=False,model_complexity=45,model_depth=20,padding_mode="zeros"):
        super(UNet,self).__init__()

        if complex:
            model_complexity = int(model_complexity // 1.414)

        self.set_size(model_complexity=model_complexity, input_channels=input_channels, model_depth=model_depth)
        self.encoders = []
        self.model_length = model_depth // 2
        
        self.ResPath_layers=[]
        
        for i in range(self.model_length):
            module = ResPath(channel = self.enc_channels[self.model_length - i])
            self.add_module("ResPath{}".format(i), module)
            self.ResPath_layers.append(module)
        
        
        for i in range(self.model_length):
            module = Encoder(self.enc_channels[i], self.enc_channels_with_rnn[i + 1], kernel_size=self.enc_kernel_sizes[i],
                             stride=self.enc_strides[i], padding=self.enc_paddings[i], complex=complex, padding_mode=padding_mode)
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []

        for i in range(self.model_length):
            module = Decoder(self.dec_channels[i] + self.enc_channels[self.model_length - i], self.dec_channels[i + 1], kernel_size=self.dec_kernel_sizes[i],
                             stride=self.dec_strides[i], padding=self.dec_paddings[i], output_padding=self.dec_output_paddings[i],complex=complex)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        self.rnns=[]
        for i in range(self.model_length):
            module = nn.LSTM(input_size=self.CD_input[i],hidden_size = self.CD_input[i],batch_first=True)
            self.add_module("rnn{}".format(i), module)
            self.rnns.append(module)


        self.fcs=[]
        for i in range(self.model_length):
            module = nn.Linear(self.CD_input[i],self.CD_input[i])
            self.add_module("fc{}".format(i), module)
            self.fcs.append(module)

        

        self.rnns_pa=[]
        for i in range(self.model_length):
            module = nn.LSTM(input_size=self.CD_input2[i],hidden_size = self.CD_input2[i],batch_first=True)
            self.add_module("rnns{}".format(i), module)
            self.rnns_pa.append(module)


        self.fcs_pa=[]
        for i in range(self.model_length):
            module = nn.Linear(self.CD_input2[i],self.CD_input2[i])
            self.add_module("fcs{}".format(i), module)
            self.fcs_pa.append(module)


        if complex:
            conv = ComplexConv2d
        else:
            conv = nn.Conv2d

        linear = conv(self.dec_channels[-1], 1, 1)

        self.add_module("linear", linear)
        self.complex = complex
        self.padding_mode = padding_mode

        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)
        self.ResPath_layers = nn.ModuleList(self.ResPath_layers)
        
        self.fc0v = nn.Linear(512,90)
        self.sigmoid = nn.Sigmoid()
        # self.up = nn.Upsample(scale_factor=4)
        
    def forward(self, video,mag):
        # pdb.set_trace()
        # video = self.up(video) #B 512 T -> B 512 4T
        
        video = video.permute(0,2,1) #[1 512 34]
        video = F.interpolate(input=video,size=(mag.size(2))) #[1 512 T]
        video_out = self.fc0v(video.permute(0,2,1)) # [B 512 T] ->[B T 321]
        # video_out = self.fc0v(video)
        #video_out,_ = self.rnn_v(video_out)
        video_out = video_out.permute(0,2,1)
        #video_out_weigth = self.sigmoid(video_out_weigth)
        #video_out = (video_out*video_out_weigth).permute(0,2,1)
        
        #video_out = self.video_encoder(video_out.permute(0,2,1)) #[1,1536,4x]
        #video_out = self.fc1v(video_out.permute(0,2,1)).permute(0,2,1)
        video_out = torch.unsqueeze(video_out,1) #[B 1 321 4T]
        
    
        amp_spec = mag
        
        # pdb.set_trace()
        amp_spec4 = torch.unsqueeze(amp_spec,1) #B 1 321 4T]
        cmp_spec = torch.cat([amp_spec4,video_out],1)
        if self.complex:
            x = cmp_spec
        else:
            x = cmp_spec
        # go down
        xs = []
        for i, encoder in enumerate(zip(self.encoders,self.rnns,self.fcs)):
            xs.append(x)
            #pdb.set_trace()
            x = encoder[0](x)

            gap_x = x.mean(1) # B Frep Time
            gap_x = gap_x.permute(0,2,1)
            gap_x,_ = encoder[1](gap_x)
            gap_x = encoder[2](gap_x)
            gap_x = gap_x.unsqueeze(1).permute(0,1,3,2)
            gap_x = self.sigmoid(gap_x)
            x = gap_x*x
        xs.append(x)
        p = x
        # pdb.set_trace()
        for i, decoder in enumerate(zip(self.decoders,self.ResPath_layers,self.rnns_pa,self.fcs_pa)):
            if i ==0:
                skipconnection = decoder[1](xs[self.model_length - i])
                gap_x = skipconnection.mean(1) # B Frep Time
                gap_x = gap_x.permute(0,2,1)
                gap_x,_ = decoder[2](gap_x)
                gap_x = decoder[3](gap_x)
                gap_x = gap_x.unsqueeze(1).permute(0,1,3,2)
                gap_x = self.sigmoid(gap_x)
                x = gap_x*skipconnection
                dec_p = decoder[0](x)
                
            # print(xs[self.model_length - 1 - i].shape)
            # print(f"p{i}, {p.shape} + x{self.model_length - 1 - i}, {xs[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")
            else:
                # pdb.set_trace()
                skipconnection = decoder[1](xs[self.model_length - i])
                gap_x = skipconnection.mean(1) # B Frep Time
                gap_x = gap_x.permute(0,2,1)
                gap_x,_ = decoder[2](gap_x)
                gap_x = decoder[3](gap_x)
                gap_x = gap_x.unsqueeze(1).permute(0,1,3,2)
                gap_x = self.sigmoid(gap_x)
                x = gap_x*skipconnection
                dec_p = torch.cat([dec_p, x], dim=1)
                dec_p = decoder[0](dec_p)
                # print(xs[self.model_length - 1 - i].shape)
                # print(f"p{i}, {p.shape} + x{self.model_length - 1 - i}, {xs[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")
            if i == self.model_length - 1:
                break
        
        #print(p.shape)
        mask = self.linear(dec_p)
        mask = self.sigmoid(mask)
        #print(mask.shape)
        
        return amp_spec*mask[:,0,:,:]

    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        if model_depth == 10:
            self.enc_channels = [input_channels,
                                 model_complexity+1,
                                 model_complexity+1,
                                 model_complexity+1,
                                 model_complexity+1,
                                 128]
            self.enc_channels_with_rnn = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity,
                                 128]

            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (7, 5),
                                     (7, 5),
                                     (5, 3),
                                     ]

            self.enc_strides = [(1, 1),
                                (1, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                ]

            self.enc_paddings = [(3, 0),
                                 (0, 3),
                                 (3, 2),
                                 (3, 2),
                                 (2, 1),
                                 ]
                              
                                 

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 ]

            self.dec_kernel_sizes = [(5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     ]

            self.dec_strides = [(2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                ]

            self.dec_paddings = [(2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 ]
            self.dec_output_paddings = [(0,0),
                                        (1,1),
                                        (0,0),
                                        (0,1),
                                        (0,0),
                                        ]
            self.CD_input = [321,
                       321,
                       161,
                       81,
                       41,
                       ]
            self.CD_output = [100,
                       100,
                       100,
                       40,
                       20,
                       ]

        elif model_depth == 20:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity*2,
                                 model_complexity*2,
                                 model_complexity*2,
                                 model_complexity*2,
                                 model_complexity*2,
                                 model_complexity*2,
                                 model_complexity*2,
                                 128]
            self.enc_channels_with_rnn = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity*2,
                                 model_complexity*2,
                                 model_complexity*2,
                                 model_complexity*2,
                                 model_complexity*2,
                                 model_complexity*2,
                                 model_complexity*2,
                                 128]

            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (7, 5),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]

            self.enc_strides = [(1, 1),
                                (1, 1),
                                (1, 1),
                                (1, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1)]

            self.enc_paddings = [(3, 0),
                                 (0, 3),
                                 (3, 2),
                                 (3, 2),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),]
                              
                                 

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2]

            self.dec_kernel_sizes = [(5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3), 
                                     (7, 5), 
                                     (7, 5), 
                                     (1, 7),
                                     (7, 1)]

            self.dec_strides = [(2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (1, 1),
                                (1, 1),
                                (1, 1),
                                (1, 1)]

            self.dec_paddings = [(2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1),
                                 (3, 2),
                                 (3, 2),
                                 (0, 3),
                                 (3, 0)]
            self.dec_output_paddings = [(0,0),#2
                                        (1,0),#4
                                        (1,0),#7
                                        (0,0),#13
                                        (0,0),#26
                                        (1,0),
                                        (0,0),
                                        (0,0),
                                        (0,0),
                                        (0,0)]
            self.CD_input = [90,
                       90,
                       90,
                       90,
                       45,
                       23,
                       12,
                       6,
                       3,
                       2]
            # self.CD_input = [321,
            #            321,
            #            161,
            #            81,
            #            41,
            #            21,
            #            11,
            #            6,
            #            3,
            #            2]
            
            # self.CD_input2 = [2,3,
            #            6,
            #            11,
            #            21,
            #            41,
            #            81,
            #            161,
            #            321,
            #            321]
            
            self.CD_input2 = [2,3,
                       6,
                       12,
                       23,
                       45,
                       90,
                       90,
                       90,
                       90]

            self.CD_output = [100,
                       100,
                       100,
                       40,
                       20,
                       10,
                       3,
                       3,
                       3,
                       2]
        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))
            
            
    # @classmethod
    # def build_video_encoder(cls,config):
    #     return Resnet_transcnn(
    #            block=config['vblock'], init_channel=config['vinit_channel'],out_channel=config['vout_channel'],
    #            blockslayers=config['vblockslayers'], stride=config['vstride'])
               
    # @classmethod
    # def build_audio_encoder(cls,config):
        
    #     return Resnet_cnn(
    #             init_channel=config['ainit_channel'],out_channel=config['aout_channel'],block=config['ablock'],
    #             blockslayers=config['ablockslayers'], stride=config['astride'])
                
    # @classmethod
    # def build_model(cls, config):
    #     video_encoder = cls.build_video_encoder(config)
    #     audio_encoder = cls.build_audio_encoder(config)
    #     return video_encoder,audio_encoder
        
def wSDRLoss(mixed, clean, clean_est, eps=2e-7):
    # Used on signal level(time-domain). Backprop-able istft should be used.
    # Batched audio inputs shape (N x T) required.
    bsum = lambda x: torch.sum(x, dim=1) # Batch preserving sum for convenience.
    def mSDRLoss(orig, est):
        # Modified SDR loss, <x, x`> / (||x|| * ||x`||) : L2 Norm.
        # Original SDR Loss: <x, x`>**2 / <x`, x`> (== ||x`||**2)
        #  > Maximize Correlation while producing minimum energy output.
        correlation = bsum(orig * est)
        energies = torch.norm(orig, p=2, dim=1) * torch.norm(est, p=2, dim=1)
        return -(correlation / (energies + eps))

    noise = mixed - clean
    noise_est = mixed - clean_est

    a = bsum(clean**2) / (bsum(clean**2) + bsum(noise**2) + eps)
    wSDR = a * mSDRLoss(clean, clean_est) + (1 - a) * mSDRLoss(noise, noise_est)
    #wSDR = mSDRLoss(clean, clean_est)
    return torch.mean(wSDR)
        
def l2_norm(s1, s2):
    #norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    #norm = torch.norm(s1*s2, 1, keepdim=True)
    
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return -torch.mean(snr)


@register_model("avse_avsr_vggtransformer_norm_DCM")
class avse_avsr_VGGTransformerModel_norm_DCM(BaseFairseqModel):
    """
    Transformers with convolutional context for ASR
    https://arxiv.org/abs/1904.11660
    """
    def __init__(self,avse,avsr):
        super().__init__()
        # pdb.set_trace()
        self.unet = avse
        #self.unet.load_state_dict(torch.load("/home/nas/user/jungwook/fairseq/examples/audio_visual_speech_enhancement/Unet/model_ckpt/AVSE_net2_m/0624/SNR0/learning_rate_0.0001_batch_10_frame_num_128/bestmodel.pth"))
        self.unet.load_state_dict(torch.load("avse_model/new_AVSE.pth"))
        for param in self.unet.parameters():
            param.half()
            param.requires_grad = False
        self.unet.eval()     
        # pdb.set_trace()
        self.avsr = avsr
        self.avsr.load_state_dict(torch.load("result/audio_visual_speech_recognition/BiModalvggtransformer_avsr_norm_DCM_base/model/new_feature_VGG_base_wi_transformer_CE_yh/checkpoint47.pt")['model'])

    def forward(self, audio_src_tokens, audio_src_lengths, video_src_tokens, video_src_lengths, prev_output_tokens, **kwargs):
        # pdb.set_trace()
        avse_out = self.unet(video_src_tokens,audio_src_tokens.permute(0,2,1))
        avse_out = avse_out.permute(0,2,1)

        audio_encoder_out = self.avsr.audio_encoder(avse_out, src_lengths=audio_src_lengths, **kwargs)
        video_encoder_out = self.avsr.video_encoder(video_src_tokens, src_lengths=video_src_lengths, **kwargs)
        
        audio_only_encoder_out = audio_encoder_out
        video_only_encoder_out = video_encoder_out
        
        av_encoder_out, av_attn = self.avsr.av_attn(
                query=audio_only_encoder_out["encoder_out"],
                key=video_only_encoder_out["encoder_out"],
                value=video_only_encoder_out["encoder_out"],
                static_kv=True,
                need_weights=True,
                )
        va_encoder_out, va_attn = self.avsr.va_attn(
                query=video_only_encoder_out["encoder_out"],
                key=audio_only_encoder_out["encoder_out"],
                value=audio_only_encoder_out["encoder_out"],
                static_kv=True,
                need_weights=True,
                )

        av_encoder_out = self.avsr.av_layer_norm(av_encoder_out)
        va_encoder_out = self.avsr.va_layer_norm(va_encoder_out)

        audio_encoder_out["encoder_out"] = av_encoder_out
        video_encoder_out["encoder_out"] = va_encoder_out
        
        decoder_out = self.avsr.decoder(
                prev_output_tokens, 
                audio_encoder_out=audio_encoder_out, 
                video_encoder_out=video_encoder_out, 
                **kwargs
                )
        
        
        return decoder_out
        
  
    def forward_ctc_encoder(self, audio_src_tokens, audio_src_lengths, video_src_tokens, video_src_lengths, prev_output_tokens, **kwargs):
        #pdb.set_trace()
        avse_out = self.unet(video_src_tokens,audio_src_tokens)
        avse_out = avse_out.permute(0,2,1)
        audio_encoder_out = self.avsr.audio_encoder(avse_out, src_lengths=audio_src_lengths, **kwargs)
        video_encoder_out = self.avsr.video_encoder(video_src_tokens, src_lengths=video_src_lengths, **kwargs)
        
        audio_only_encoder_out = audio_encoder_out
        video_only_encoder_out = video_encoder_out
        
        av_encoder_out, av_attn = self.avsr.av_attn(
                query=audio_only_encoder_out["encoder_out"],
                key=video_only_encoder_out["encoder_out"],
                value=video_only_encoder_out["encoder_out"],
                static_kv=True,
                need_weights=True,
                )
        va_encoder_out, va_attn = self.avsr.va_attn(
                query=video_only_encoder_out["encoder_out"],
                key=audio_only_encoder_out["encoder_out"],
                value=audio_only_encoder_out["encoder_out"],
                static_kv=True,
                need_weights=True,
                )

        av_encoder_out = self.avsr.av_layer_norm(av_encoder_out)
        va_encoder_out = self.avsr.va_layer_norm(va_encoder_out)
    
        # upsampleing video_encoder_out to audio_encoder_output's time resolution
        # [T, B, F] -> [B, F, T]
        va_encoder_out = va_encoder_out.permute(1,2,0)
        va_encoder_out = torch.nn.functional.interpolate(
                va_encoder_out,
                av_encoder_out.size(0)
                )
        # [B, F, T] -> [T, B, F]
        va_encoder_out = va_encoder_out.permute(2,0,1)
        # concat audio and video encoder output and feed to fc
        fusion_encoder_out = self.avsr.ctc_fusion_layer(
                torch.cat(
                    (av_encoder_out, va_encoder_out),2
                    )
                )
        audio_encoder_out["encoder_out"] = fusion_encoder_out
        return audio_encoder_out

    def forward_ctc_decoder(self, prev_output_tokens, audio_encoder_outs, video_encoder_outs, **kwargs):
        #pdb.set_trace()
        av_encoder_out = audio_encoder_outs["encoder_out"] 
        va_encoder_out = video_encoder_outs["encoder_out"] 
        # upsampleing video_encoder_out to audio_encoder_output's time resolution
        # [T, B, F] -> [B, F, T]
        va_encoder_out = va_encoder_out.permute(1,2,0)
        va_encoder_out = torch.nn.functional.interpolate(
                va_encoder_out,
                av_encoder_out.size(0)
                )
        # [B, F, T] -> [T, B, F]
        va_encoder_out = va_encoder_out.permute(2,0,1)
        # concat audio and video encoder output and feed to fc
        fusion_encoder_out = self.avsr.ctc_fusion_layer(
                torch.cat(
                    (av_encoder_out, va_encoder_out),2
                    )
                )
        # [T, B, C] -> [B, T, C]
        fusion_encoder_out = torch.nn.functional.log_softmax(fusion_encoder_out.permute(1,0,2),dim=2)
        #fusion_encoder_out = fusion_encoder_out.permute(1,0,2)

        return fusion_encoder_out
        
    def forward_decoder(self, prev_output_tokens, audio_encoder_outs, video_encoder_outs, incremental_state, **kwargs):
        #pdb.set_trace()
        audio_only_encoder_out = audio_encoder_outs
        video_only_encoder_out = video_encoder_outs
        
        av_encoder_out, av_attn = self.avsr.av_attn(
                query=audio_only_encoder_out["encoder_out"],
                key=video_only_encoder_out["encoder_out"],
                value=video_only_encoder_out["encoder_out"],
                static_kv=True,
                need_weights=True,
                )
        va_encoder_out, va_attn = self.avsr.va_attn(
                query=video_only_encoder_out["encoder_out"],
                key=audio_only_encoder_out["encoder_out"],
                value=audio_only_encoder_out["encoder_out"],
                static_kv=True,
                need_weights=True,
                )

        av_encoder_out = self.avsr.av_layer_norm(av_encoder_out)
        va_encoder_out = self.avsr.va_layer_norm(va_encoder_out)

        audio_encoder_outs["encoder_out"] = av_encoder_out
        video_encoder_outs["encoder_out"] = va_encoder_out
        
        lprobs_att, avg_attn_scores = self.avsr.decoder(prev_output_tokens, audio_encoder_outs, video_encoder_outs, incremental_state, **kwargs)
        lprobs_att = torch.nn.functional.log_softmax(lprobs_att, dim=2)
        lprobs_ctc = self.avsr.forward_ctc_decoder(prev_output_tokens, audio_encoder_outs, video_encoder_outs, **kwargs)
        lprobs = 0.9 * lprobs_att + 0.1 * lprobs_ctc
        return lprobs, avg_attn_scores
    
    def extract_features(self, audio_src_tokens, audio_src_lengths, video_src_tokens, video_src_lengths, prev_output_tokens, **kwargs):
        avse_out = self.unet(video_src_tokens.permute(0,2,1),audio_src_tokens)
        avse_out = avse_out.permute(0,2,1)
        audio_encoder_out = self.avsr.audio_encoder(avse_out, src_lengths=audio_src_lengths, **kwargs)
        video_encoder_out = self.avsr.video_encoder(video_src_tokens, src_lengths=video_src_lengths, **kwargs)
        features = self.avsr.decoder.extract_features(prev_output_tokens, audio_encoder_out=audio_encoder_out, video_encoder_out=video_encoder_out, **kwargs)
        return features
 
    def output_layer(self, features, **kwargs):
        return self.avsr.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        return (self.avsr.audio_encoder.max_positions(), self.avsr.video_encoder.max_positions(), self.avsr.decoder.max_positions())

    def max_decoder_positions(self):
        return self.avsr.decoder.max_positions()
    
    @classmethod
    def build_avsr_audio_encoder(cls, args, task):
        return VGGTransformerEncoder(
            input_feat_per_channel=args.audio_input_feat_per_channel,
            vggblock_config=eval(args.audio_vggblock_enc_config),
            transformer_config=eval(args.transformer_enc_config),
            encoder_output_dim=args.audio_enc_output_dim,
            in_channels=args.in_channels,
        )
    
    @classmethod
    def build_avsr_video_encoder(cls, args, task):
        return VGGTransformerEncoder(
            input_feat_per_channel=args.video_input_feat_per_channel,
            vggblock_config=eval(args.video_vggblock_enc_config),
            transformer_config=eval(args.transformer_enc_config),
            encoder_output_dim=args.video_enc_output_dim,
            in_channels=args.in_channels,
        )

    @classmethod
    def build_avsr_decoder(cls, args, task):
        return TransformerBiModalDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.tgt_embed_dim,
            transformer_config=eval(args.transformer_dec_config),
            conv_config=eval(args.conv_dec_config),
            encoder_output_dim=args.enc_output_dim,
        )

    @classmethod
    def build_avsr_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        base_architecture(args)

        audio_encoder = cls.build_avsr_audio_encoder(args, task)
        video_encoder = cls.build_avsr_video_encoder(args, task)
        decoder = cls.build_avsr_decoder(args, task)
        return VGGTransformerModel_norm_DCM(
            audio_encoder = audio_encoder,
            video_encoder = video_encoder,
            decoder = decoder)
    
    @classmethod
    def build_avse_model(cls, args, task):
    
        return UNet()
    
    @classmethod
    def build_model(cls, args, task):
        avsr = cls.build_avsr_model(args,task)
        avse = cls.build_avse_model(args,task)
        # pdb.set_trace()
        return cls(avse,avsr)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        # pdb.set_trace()
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs


class VGGTransformerModel_norm_DCM(BaseFairseqModel):
    """
    Transformers with convolutional context for ASR
    https://arxiv.org/abs/1904.11660
    """
    def __init__(self, audio_encoder, video_encoder, decoder):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
    
        # embeded_dim
        # num_head
        # kdim
        # vdim
        # dropout
        self.av_attn = MultiheadAttention(
                512,
                8,
                512,
                512,
                0.15,
                encoder_decoder_attention=True
                ) 
        self.va_attn = MultiheadAttention(
                512,
                8,
                512,
                512,
                0.15,
                encoder_decoder_attention=True
                )
        self.av_layer_norm = LayerNorm(512)
        self.va_layer_norm = LayerNorm(512)
        self.ctc_fusion_layer = Linear(1024, 44) # fusion audio and video information to target text sequence

        self.decoder = decoder


    def forward(self, audio_src_tokens, audio_src_lengths, video_src_tokens, video_src_lengths, prev_output_tokens, **kwargs):
        #pdb.set_trace()
        audio_encoder_out = self.audio_encoder(audio_src_tokens, src_lengths=audio_src_lengths, **kwargs)
        video_encoder_out = self.video_encoder(video_src_tokens, src_lengths=video_src_lengths, **kwargs)
        
        audio_only_encoder_out = audio_encoder_out
        video_only_encoder_out = video_encoder_out
        
        av_encoder_out, av_attn = self.av_attn(
                query=audio_only_encoder_out["encoder_out"],
                key=video_only_encoder_out["encoder_out"],
                value=video_only_encoder_out["encoder_out"],
                static_kv=True,
                need_weights=True,
                )
        va_encoder_out, va_attn = self.va_attn(
                query=video_only_encoder_out["encoder_out"],
                key=audio_only_encoder_out["encoder_out"],
                value=audio_only_encoder_out["encoder_out"],
                static_kv=True,
                need_weights=True,
                )

        av_encoder_out = self.av_layer_norm(av_encoder_out)
        va_encoder_out = self.va_layer_norm(va_encoder_out)

        audio_encoder_out["encoder_out"] = av_encoder_out
        video_encoder_out["encoder_out"] = va_encoder_out
        
        decoder_out = self.decoder(
                prev_output_tokens, 
                audio_encoder_out=audio_encoder_out, 
                video_encoder_out=video_encoder_out, 
                **kwargs
                )
        
        #####################################################
#        import os
#        import scipy.io as sio
#        path = "./img/"
#        filename = "dcm_av_attn"
#        i = 0
#        while os.path.exists(path + filename + "_" + str(i) + ".mat"):
#            i += 1
#        filename = filename + '_' + str(i)
#        sio.savemat(path + filename + ".mat", {filename:av_attn[0,:,:].cpu().detach().numpy()})
#        filename = "dcm_va_attn"
#        i = 0
#        while os.path.exists(path + filename + "_" + str(i) + ".mat"):
#            i += 1
#        filename = filename + '_' + str(i)
#        sio.savemat(path + filename + ".mat", {filename:va_attn[0,:,:].cpu().detach().numpy()})
        #####################################################

        return decoder_out
     
    def forward_ctc_encoder(self, audio_src_tokens, audio_src_lengths, video_src_tokens, video_src_lengths, prev_output_tokens, **kwargs):
        #pdb.set_trace()
        audio_encoder_out = self.audio_encoder(audio_src_tokens, src_lengths=audio_src_lengths, **kwargs)
        video_encoder_out = self.video_encoder(video_src_tokens, src_lengths=video_src_lengths, **kwargs)
        
        audio_only_encoder_out = audio_encoder_out
        video_only_encoder_out = video_encoder_out
        
        av_encoder_out, av_attn = self.av_attn(
                query=audio_only_encoder_out["encoder_out"],
                key=video_only_encoder_out["encoder_out"],
                value=video_only_encoder_out["encoder_out"],
                static_kv=True,
                need_weights=True,
                )
        va_encoder_out, va_attn = self.va_attn(
                query=video_only_encoder_out["encoder_out"],
                key=audio_only_encoder_out["encoder_out"],
                value=audio_only_encoder_out["encoder_out"],
                static_kv=True,
                need_weights=True,
                )

        av_encoder_out = self.av_layer_norm(av_encoder_out)
        va_encoder_out = self.va_layer_norm(va_encoder_out)
    
        # upsampleing video_encoder_out to audio_encoder_output's time resolution
        # [T, B, F] -> [B, F, T]
        va_encoder_out = va_encoder_out.permute(1,2,0)
        va_encoder_out = torch.nn.functional.interpolate(
                va_encoder_out,
                av_encoder_out.size(0)
                )
        # [B, F, T] -> [T, B, F]
        va_encoder_out = va_encoder_out.permute(2,0,1)
        # concat audio and video encoder output and feed to fc
        fusion_encoder_out = self.ctc_fusion_layer(
                torch.cat(
                    (av_encoder_out, va_encoder_out),2
                    )
                )
        audio_encoder_out["encoder_out"] = fusion_encoder_out
        return audio_encoder_out
 
    def forward_ctc_decoder(self, prev_output_tokens, audio_encoder_outs, video_encoder_outs, **kwargs):
        #pdb.set_trace()
        av_encoder_out = audio_encoder_outs["encoder_out"] 
        va_encoder_out = video_encoder_outs["encoder_out"] 
        # upsampleing video_encoder_out to audio_encoder_output's time resolution
        # [T, B, F] -> [B, F, T]
        va_encoder_out = va_encoder_out.permute(1,2,0)
        va_encoder_out = torch.nn.functional.interpolate(
                va_encoder_out,
                av_encoder_out.size(0)
                )
        # [B, F, T] -> [T, B, F]
        va_encoder_out = va_encoder_out.permute(2,0,1)
        # concat audio and video encoder output and feed to fc
        fusion_encoder_out = self.ctc_fusion_layer(
                torch.cat(
                    (av_encoder_out, va_encoder_out),2
                    )
                )
        # [T, B, C] -> [B, T, C]
        fusion_encoder_out = torch.nn.functional.log_softmax(fusion_encoder_out.permute(1,0,2),dim=2)
        #fusion_encoder_out = fusion_encoder_out.permute(1,0,2)

        return fusion_encoder_out
  
    def forward_decoder(self, prev_output_tokens, audio_encoder_outs, video_encoder_outs, incremental_state, **kwargs):
        #pdb.set_trace()
        audio_only_encoder_out = audio_encoder_outs
        video_only_encoder_out = video_encoder_outs
        
        av_encoder_out, av_attn = self.av_attn(
                query=audio_only_encoder_out["encoder_out"],
                key=video_only_encoder_out["encoder_out"],
                value=video_only_encoder_out["encoder_out"],
                static_kv=True,
                need_weights=True,
                )
        va_encoder_out, va_attn = self.va_attn(
                query=video_only_encoder_out["encoder_out"],
                key=audio_only_encoder_out["encoder_out"],
                value=audio_only_encoder_out["encoder_out"],
                static_kv=True,
                need_weights=True,
                )

        av_encoder_out = self.av_layer_norm(av_encoder_out)
        va_encoder_out = self.va_layer_norm(va_encoder_out)

        audio_encoder_outs["encoder_out"] = av_encoder_out
        video_encoder_outs["encoder_out"] = va_encoder_out
        
        lprobs_att, avg_attn_scores = self.decoder(prev_output_tokens, audio_encoder_outs, video_encoder_outs, incremental_state, **kwargs)
        lprobs_att = torch.nn.functional.log_softmax(lprobs_att, dim=2)
        lprobs_ctc = self.forward_ctc_decoder(prev_output_tokens, audio_encoder_outs, video_encoder_outs, **kwargs)
        lprobs = 0.9 * lprobs_att + 0.1 * lprobs_ctc
        return lprobs, avg_attn_scores
    
    def extract_features(self, audio_src_tokens, audio_src_lengths, video_src_tokens, video_src_lengths, prev_output_tokens, **kwargs):
        audio_encoder_out = self.audio_encoder(audio_src_tokens, src_lengths=audio_src_lengths, **kwargs)
        video_encoder_out = self.video_encoder(video_src_tokens, src_lengths=video_src_lengths, **kwargs)
        features = self.decoder.extract_features(prev_output_tokens, audio_encoder_out=audio_encoder_out, video_encoder_out=video_encoder_out, **kwargs)
        return features
 
    def output_layer(self, features, **kwargs):
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        return (self.audio_encoder.max_positions(), self.video_encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        return self.decoder.max_positions()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--input-feat-per-channel",
            type=int,
            metavar="N",
            help="encoder input dimension per input channel",
        )
        parser.add_argument(
            "--vggblock-enc-config",
            type=str,
            metavar="EXPR",
            help="""
    an array of tuples each containing the configuration of one vggblock:
    [(out_channels,
      conv_kernel_size,
      pooling_kernel_size,
      num_conv_layers,
      use_layer_norm), ...])
            """,
        )
        parser.add_argument(
            "--transformer-enc-config",
            type=str,
            metavar="EXPR",
            help=""""
    a tuple containing the configuration of the encoder transformer layers
    configurations:
    [(input_dim,
      num_heads,
      ffn_dim,
      normalize_before,
      dropout,
      attention_dropout,
      relu_dropout), ...]')
            """,
        )
        parser.add_argument(
            "--enc-output-dim",
            type=int,
            metavar="N",
            help="""
    encoder output dimension, can be None. If specified, projecting the
    transformer output to the specified dimension""",
        )
        parser.add_argument(
            "--in-channels",
            type=int,
            metavar="N",
            help="number of encoder input channels",
        )
        parser.add_argument(
            "--tgt-embed-dim",
            type=int,
            metavar="N",
            help="embedding dimension of the decoder target tokens",
        )
        parser.add_argument(
            "--transformer-dec-config",
            type=str,
            metavar="EXPR",
            help="""
    a tuple containing the configuration of the decoder transformer layers
    configurations:
    [(input_dim,
      num_heads,
      ffn_dim,
      normalize_before,
      dropout,
      attention_dropout,
      relu_dropout), ...]
            """,
        )
        parser.add_argument(
            "--conv-dec-config",
            type=str,
            metavar="EXPR",
            help="""
    an array of tuples for the decoder 1-D convolution config
        [(out_channels, conv_kernel_size, use_layer_norm), ...]""",
        )

    @classmethod
    def build_audio_encoder(cls, args, task):
        return VGGTransformerEncoder(
            input_feat_per_channel=args.audio_input_feat_per_channel,
            vggblock_config=eval(args.audio_vggblock_enc_config),
            transformer_config=eval(args.transformer_enc_config),
            encoder_output_dim=args.audio_enc_output_dim,
            in_channels=args.in_channels,
        )
    
    @classmethod
    def build_video_encoder(cls, args, task):
        return VGGTransformerEncoder(
            input_feat_per_channel=args.video_input_feat_per_channel,
            vggblock_config=eval(args.video_vggblock_enc_config),
            transformer_config=eval(args.transformer_enc_config),
            encoder_output_dim=args.video_enc_output_dim,
            in_channels=args.in_channels,
        )

    @classmethod
    def build_decoder(cls, args, task):
        return TransformerBiModalDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.tgt_embed_dim,
            transformer_config=eval(args.transformer_dec_config),
            conv_config=eval(args.conv_dec_config),
            encoder_output_dim=args.enc_output_dim,
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        base_architecture(args)

        audio_encoder = cls.build_audio_encoder(args, task)
        video_encoder = cls.build_video_encoder(args, task)
        decoder = cls.build_decoder(args, task)
        return cls(audio_encoder, video_encoder, decoder)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = super().get_normalized_probs(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

DEFAULT_ENC_VGGBLOCK_CONFIG = ((32, 3, 2, 2, False),) * 2
DEFAULT_ENC_TRANSFORMER_CONFIG = ((256, 4, 1024, True, 0.2, 0.2, 0.2),) * 2
# 256: embedding dimension
# 4: number of heads
# 1024: FFN
# True: apply layerNorm before (dropout + resiaul) instead of after
# 0.2 (dropout): dropout after MultiheadAttention and second FC
# 0.2 (attention_dropout): dropout in MultiheadAttention
# 0.2 (relu_dropout): dropout after ReLu
DEFAULT_DEC_TRANSFORMER_CONFIG = ((256, 2, 1024, True, 0.2, 0.2, 0.2),) * 2
DEFAULT_DEC_CONV_CONFIG = ((256, 3, True),) * 2


# TODO: repace transformer encoder config from one liner
# to explicit args to get rid of this transformation
def prepare_transformer_encoder_params(
    input_dim,
    num_heads,
    ffn_dim,
    normalize_before,
    dropout,
    attention_dropout,
    relu_dropout,
):
    args = argparse.Namespace()
    args.encoder_embed_dim = input_dim
    args.encoder_attention_heads = num_heads
    args.attention_dropout = attention_dropout
    args.dropout = dropout
    args.activation_dropout = relu_dropout
    args.encoder_normalize_before = normalize_before
    args.encoder_ffn_embed_dim = ffn_dim
    return args


def prepare_transformer_decoder_params(
    input_dim,
    num_heads,
    ffn_dim,
    normalize_before,
    dropout,
    attention_dropout,
    relu_dropout,
):
    args = argparse.Namespace()
    args.decoder_embed_dim = input_dim
    args.decoder_attention_heads = num_heads
    args.attention_dropout = attention_dropout
    args.dropout = dropout
    args.activation_dropout = relu_dropout
    args.decoder_normalize_before = normalize_before
    args.decoder_ffn_embed_dim = ffn_dim
    return args


class VGGTransformerEncoder(FairseqEncoder):
    """VGG + Transformer encoder"""

    def __init__(
        self,
        input_feat_per_channel,
        vggblock_config=DEFAULT_ENC_VGGBLOCK_CONFIG,
        transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG,
        encoder_output_dim=512,
        in_channels=1,
        transformer_context=None,
        transformer_sampling=None,
        modality=None,
    ):
        """constructor for VGGTransformerEncoder

        Args:
            - input_feat_per_channel: feature dim (not including stacked,
              just base feature)
            - in_channel: # input channels (e.g., if stack 8 feature vector
                together, this is 8)
            - vggblock_config: configuration of vggblock, see comments on
                DEFAULT_ENC_VGGBLOCK_CONFIG
            - transformer_config: configuration of transformer layer, see comments
                on DEFAULT_ENC_TRANSFORMER_CONFIG
            - encoder_output_dim: final transformer output embedding dimension
            - transformer_context: (left, right) if set, self-attention will be focused
              on (t-left, t+right)
            - transformer_sampling: an iterable of int, must match with
              len(transformer_config), transformer_sampling[i] indicates sampling
              factor for i-th transformer layer, after multihead att and feedfoward
              part
        """
        super().__init__(None)

        self.num_vggblocks = 0
        if vggblock_config is not None:
            if not isinstance(vggblock_config, Iterable):
                raise ValueError("vggblock_config is not iterable")
            self.num_vggblocks = len(vggblock_config)

        self.conv_layers = nn.ModuleList()
        self.in_channels = in_channels
        self.input_dim = input_feat_per_channel

        if vggblock_config is not None:
            for _, config in enumerate(vggblock_config):
                (
                    out_channels,
                    conv_kernel_size,
                    pooling_kernel_size,
                    num_conv_layers,
                    layer_norm,
                ) = config
                self.conv_layers.append(
                    VGGBlock(
                        in_channels,
                        out_channels,
                        conv_kernel_size,
                        pooling_kernel_size,
                        num_conv_layers,
                        input_dim=input_feat_per_channel,
                        layer_norm=layer_norm,
                    )
                )
                in_channels = out_channels
                input_feat_per_channel = self.conv_layers[-1].output_dim

        transformer_input_dim = self.infer_conv_output_dim(
            self.in_channels, self.input_dim
        )
        # transformer_input_dim is the output dimension of VGG part

        self.validate_transformer_config(transformer_config)
        self.transformer_context = self.parse_transformer_context(transformer_context)
        self.transformer_sampling = self.parse_transformer_sampling(
            transformer_sampling, len(transformer_config)
        )

        self.transformer_layers = nn.ModuleList()

        if transformer_input_dim != transformer_config[0][0]:
            self.transformer_layers.append(
                Linear(transformer_input_dim, transformer_config[0][0])
            )
        self.transformer_layers.append(
            TransformerEncoderLayer(
                prepare_transformer_encoder_params(*transformer_config[0])
            )
        )

        for i in range(1, len(transformer_config)):
            if transformer_config[i - 1][0] != transformer_config[i][0]:
                self.transformer_layers.append(
                    Linear(transformer_config[i - 1][0], transformer_config[i][0])
                )
            self.transformer_layers.append(
                TransformerEncoderLayer(
                    prepare_transformer_encoder_params(*transformer_config[i])
                )
            )

        self.encoder_output_dim = encoder_output_dim
        self.transformer_layers.extend(
            [
                Linear(transformer_config[-1][0], encoder_output_dim),
                LayerNorm(encoder_output_dim),
            ]
        )

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        src_tokens: padded tensor (B, T, C * feat)
        src_lengths: tensor of original lengths of input utterances (B,)
        """
        # pdb.set_trace()
        bsz, max_seq_len, _ = src_tokens.size()
        x = src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
        x = x.transpose(1, 2).contiguous()
        # (B, C, T, feat)

        for layer_idx in range(len(self.conv_layers)):
            x = self.conv_layers[layer_idx](x)

        bsz, _, output_seq_len, _ = x.size()

        # (B, C, T, feat) -> (B, T, C, feat) -> (T, B, C, feat) -> (T, B, C * feat)
        x = x.transpose(1, 2).transpose(0, 1)
        x = x.contiguous().view(output_seq_len, bsz, -1)

        subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
        # TODO: shouldn't subsampling_factor determined in advance ?
        input_lengths = (src_lengths.float() / subsampling_factor).ceil().long()
        input_lengths = torch.clamp(input_lengths, min=0, max=x.size(0))
        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            input_lengths, batch_first=True
        )
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        attn_mask = self.lengths_to_attn_mask(input_lengths, subsampling_factor)

        transformer_layer_idx = 0

        for layer_idx in range(len(self.transformer_layers)):

            if isinstance(self.transformer_layers[layer_idx], TransformerEncoderLayer):
                x = self.transformer_layers[layer_idx](
                    x, encoder_padding_mask, attn_mask
                )

                if self.transformer_sampling[transformer_layer_idx] != 1:
                    sampling_factor = self.transformer_sampling[transformer_layer_idx]
                    x, encoder_padding_mask, attn_mask = self.slice(
                        x, encoder_padding_mask, attn_mask, sampling_factor
                    )

                transformer_layer_idx += 1

            else:
                x = self.transformer_layers[layer_idx](x)

        # encoder_padding_maks is a (T x B) tensor, its [t, b] elements indicate
        # whether encoder_output[t, b] is valid or not (valid=0, invalid=1)
        # pdb.set_trace()
        return {
            "encoder_out": x,  # (T, B, C)
            "encoder_padding_mask": encoder_padding_mask.t()
            if encoder_padding_mask is not None
            else None,
            # (B, T) --> (T, B)
        }

    def infer_conv_output_dim(self, in_channels, input_dim):
        sample_seq_len = 200
        sample_bsz = 10
        x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
        for i, _ in enumerate(self.conv_layers):
            x = self.conv_layers[i](x)
        x = x.transpose(1, 2)
        mb, seq = x.size()[:2]
        return x.contiguous().view(mb, seq, -1).size(-1)

    def validate_transformer_config(self, transformer_config):
        for config in transformer_config:
            input_dim, num_heads = config[:2]
            if input_dim % num_heads != 0:
                msg = (
                    "ERROR in transformer config {}:".format(config)
                    + "input dimension {} ".format(input_dim)
                    + "not dividable by number of heads".format(num_heads)
                )
                raise ValueError(msg)

    def parse_transformer_context(self, transformer_context):
        """
        transformer_context can be the following:
        -   None; indicates no context is used, i.e.,
            transformer can access full context
        -   a tuple/list of two int; indicates left and right context,
            any number <0 indicates infinite context
                * e.g., (5, 6) indicates that for query at x_t, transformer can
                access [t-5, t+6] (inclusive)
                * e.g., (-1, 6) indicates that for query at x_t, transformer can
                access [0, t+6] (inclusive)
        """
        if transformer_context is None:
            return None

        if not isinstance(transformer_context, Iterable):
            raise ValueError("transformer context must be Iterable if it is not None")

        if len(transformer_context) != 2:
            raise ValueError("transformer context must have length 2")

        left_context = transformer_context[0]
        if left_context < 0:
            left_context = None

        right_context = transformer_context[1]
        if right_context < 0:
            right_context = None

        if left_context is None and right_context is None:
            return None

        return (left_context, right_context)

    def parse_transformer_sampling(self, transformer_sampling, num_layers):
        """
        parsing transformer sampling configuration

        Args:
            - transformer_sampling, accepted input:
                * None, indicating no sampling
                * an Iterable with int (>0) as element
            - num_layers, expected number of transformer layers, must match with
              the length of transformer_sampling if it is not None

        Returns:
            - A tuple with length num_layers
        """
        if transformer_sampling is None:
            return (1,) * num_layers

        if not isinstance(transformer_sampling, Iterable):
            raise ValueError(
                "transformer_sampling must be an iterable if it is not None"
            )

        if len(transformer_sampling) != num_layers:
            raise ValueError(
                "transformer_sampling {} does not match with the number "
                + "of layers {}".format(transformer_sampling, num_layers)
            )

        for layer, value in enumerate(transformer_sampling):
            if not isinstance(value, int):
                raise ValueError("Invalid value in transformer_sampling: ")
            if value < 1:
                raise ValueError(
                    "{} layer's subsampling is {}.".format(layer, value)
                    + " This is not allowed! "
                )
        return transformer_sampling

    def slice(self, embedding, padding_mask, attn_mask, sampling_factor):
        """
        embedding is a (T, B, D) tensor
        padding_mask is a (B, T) tensor or None
        attn_mask is a (T, T) tensor or None
        """
        embedding = embedding[::sampling_factor, :, :]
        if padding_mask is not None:
            padding_mask = padding_mask[:, ::sampling_factor]
        if attn_mask is not None:
            attn_mask = attn_mask[::sampling_factor, ::sampling_factor]

        return embedding, padding_mask, attn_mask

    def lengths_to_attn_mask(self, input_lengths, subsampling_factor=1):
        """
        create attention mask according to sequence lengths and transformer
        context

        Args:
            - input_lengths: (B, )-shape Int/Long tensor; input_lengths[b] is
              the length of b-th sequence
            - subsampling_factor: int
                * Note that the left_context and right_context is specified in
                  the input frame-level while input to transformer may already
                  go through subsampling (e.g., the use of striding in vggblock)
                  we use subsampling_factor to scale the left/right context

        Return:
            - a (T, T) binary tensor or None, where T is max(input_lengths)
                * if self.transformer_context is None, None
                * if left_context is None,
                    * attn_mask[t, t + right_context + 1:] = 1
                    * others = 0
                * if right_context is None,
                    * attn_mask[t, 0:t - left_context] = 1
                    * others = 0
                * elsif
                    * attn_mask[t, t - left_context: t + right_context + 1] = 0
                    * others = 1
        """
        if self.transformer_context is None:
            return None

        maxT = torch.max(input_lengths).item()
        attn_mask = torch.zeros(maxT, maxT)

        left_context = self.transformer_context[0]
        right_context = self.transformer_context[1]
        if left_context is not None:
            left_context = math.ceil(self.transformer_context[0] / subsampling_factor)
        if right_context is not None:
            right_context = math.ceil(self.transformer_context[1] / subsampling_factor)

        for t in range(maxT):
            if left_context is not None:
                st = 0
                en = max(st, t - left_context)
                attn_mask[t, st:en] = 1
            if right_context is not None:
                st = t + right_context + 1
                st = min(st, maxT - 1)
                attn_mask[t, st:] = 1

        return attn_mask.to(input_lengths.device)

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
            1, new_order
        )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(1, new_order)
        return encoder_out


class TransformerBiModalDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerBiModalDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(
        self,
        dictionary,
        embed_dim=512,
        transformer_config=DEFAULT_ENC_TRANSFORMER_CONFIG,
        conv_config=DEFAULT_DEC_CONV_CONFIG,
        encoder_output_dim=512,
    ):

        super().__init__(dictionary)
        vocab_size = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(vocab_size, embed_dim, self.padding_idx)

        self.conv_layers = nn.ModuleList()
        if conv_config is not None:
            for i in range(len(conv_config)):
                out_channels, kernel_size, layer_norm = conv_config[i]
                if i == 0:
                    conv_layer = LinearizedConv1d(
                        embed_dim, out_channels, kernel_size, padding=kernel_size - 1
                    )
                else:
                    conv_layer = LinearizedConv1d(
                        conv_config[i - 1][0],
                        out_channels,
                        kernel_size,
                        padding=kernel_size - 1,
                    )
                self.conv_layers.append(conv_layer)
                if layer_norm:
                    self.conv_layers.append(nn.LayerNorm(out_channels))
                self.conv_layers.append(nn.ReLU())

        self.layers = nn.ModuleList()
        if conv_config is not None:
            if conv_config[-1][0] != transformer_config[0][0]:
                self.layers.append(Linear(conv_config[-1][0], transformer_config[0][0]))
        self.layers.append(TransformerBiModalDecoderLayer(
            prepare_transformer_decoder_params(*transformer_config[0])
        ))

        for i in range(1, len(transformer_config)):
            if transformer_config[i - 1][0] != transformer_config[i][0]:
                self.layers.append(
                    Linear(transformer_config[i - 1][0], transformer_config[i][0])
                )
            self.layers.append(TransformerBiModalDecoderLayer(
                prepare_transformer_decoder_params(*transformer_config[i])
            ))
        self.fc_out = Linear(transformer_config[-1][0], vocab_size)
    
    def extract_features(self, prev_output_tokens, audio_encoder_out=None, video_encoder_out=None, incremental_state=None):
        raise NotImplementedError

    def forward(self, prev_output_tokens, audio_encoder_out=None, video_encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            audio_encoder_out or video_encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        target_padding_mask = (
            (prev_output_tokens == self.padding_idx).to(prev_output_tokens.device)
            if incremental_state is None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)
        for layer in self.conv_layers:
            if isinstance(layer, LinearizedConvolution):
                x = layer(x, incremental_state)
            else:
                x = layer(x)

        # B x T x C -> T x B x C
        x = self._transpose_if_inference(x, incremental_state)
        # decoder layers
        for layer in self.layers:
            if isinstance(layer, TransformerBiModalDecoderLayer):
                x, _ = layer(
                    x,
                    (audio_encoder_out["encoder_out"] if audio_encoder_out is not None else None),
                    (video_encoder_out["encoder_out"] if video_encoder_out is not None else None),
                    (
                        audio_encoder_out["encoder_padding_mask"].t()
                        if audio_encoder_out["encoder_padding_mask"] is not None
                        else None
                    ),
                    (
                        video_encoder_out["encoder_padding_mask"].t()
                        if video_encoder_out["encoder_padding_mask"] is not None
                        else None
                    ),
                    incremental_state,
                    self_attn_mask=(
                        self.buffered_future_mask(x)
                        if incremental_state is None
                        else None
                    ),
                    self_attn_padding_mask=(
                        target_padding_mask if incremental_state is None else None
                    ),
                )
            else:
                x = layer(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = self.fc_out(x)
        return x, None

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def _transpose_if_training(self, x, incremental_state):
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x

    def _transpose_if_inference(self, x, incremental_state):
        if incremental_state is not None:
            x = x.transpose(0, 1)
        return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    # nn.init.uniform_(m.weight, -0.1, 0.1)
    # nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    # m.weight.data.uniform_(-0.1, 0.1)
    # if bias:
    #     m.bias.data.uniform_(-0.1, 0.1)
    return m


def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


# seq2seq models
def base_architecture(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 40)
    args.vggblock_enc_config = getattr(
        args, "vggblock_enc_config", DEFAULT_ENC_VGGBLOCK_CONFIG
    )
    args.transformer_enc_config = getattr(
        args, "transformer_enc_config", DEFAULT_ENC_TRANSFORMER_CONFIG
    )
    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
    args.in_channels = getattr(args, "in_channels", 1)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 128)
    args.transformer_dec_config = getattr(
        args, "transformer_dec_config", DEFAULT_ENC_TRANSFORMER_CONFIG
    )
    args.conv_dec_config = getattr(args, "conv_dec_config", DEFAULT_DEC_CONV_CONFIG)
    args.transformer_context = getattr(args, "transformer_context", "None")

#
#@register_model_architecture("asr_vggtransformer", "vggtransformer_1")
#def vggtransformer_1(args):
#    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
#    args.vggblock_enc_config = getattr(
#        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
#    )
#    args.transformer_enc_config = getattr(
#        args,
#        "transformer_enc_config",
#        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 14",
#    )
#    args.enc_output_dim = getattr(args, "enc_output_dim", 1024)
#    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 128)
#    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
#    args.transformer_dec_config = getattr(
#        args,
#        "transformer_dec_config",
#        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 4",
#    )
#
#
#@register_model_architecture("asr_vggtransformer", "vggtransformer_2")
#def vggtransformer_2(args):
#    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
#    args.vggblock_enc_config = getattr(
#        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
#    )
#    args.transformer_enc_config = getattr(
#        args,
#        "transformer_enc_config",
#        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 16",
#    )
#    args.enc_output_dim = getattr(args, "enc_output_dim", 1024)
#    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
#    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
#    args.transformer_dec_config = getattr(
#        args,
#        "transformer_dec_config",
#        "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 6",
#    )
#
#
#@register_model_architecture("asr_vggtransformer", "vggtransformer_base")
#def vggtransformer_base(args):
#    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
#    args.vggblock_enc_config = getattr(
#        args, "vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
#    )
#    args.transformer_enc_config = getattr(
#        args, "transformer_enc_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 12"
#    )
#
#    args.enc_output_dim = getattr(args, "enc_output_dim", 512)
#    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
#    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
#    args.transformer_dec_config = getattr(
#        args, "transformer_dec_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 6"
#    )
#    # Size estimations:
#    # Encoder:
#    #   - vggblock param: 64*1*3*3 + 64*64*3*3 + 128*64*3*3  + 128*128*3 = 258K
#    #   Transformer:
#    #   - input dimension adapter: 2560 x 512 -> 1.31M
#    #   - transformer_layers (x12) --> 37.74M
#    #       * MultiheadAttention: 512*512*3 (in_proj) + 512*512 (out_proj) = 1.048M
#    #       * FFN weight: 512*2048*2 = 2.097M
#    #   - output dimension adapter: 512 x 512 -> 0.26 M
#    # Decoder:
#    #   - LinearizedConv1d: 512 * 256 * 3 + 256 * 256 * 3 * 3
#    #   - transformer_layer: (x6) --> 25.16M
#    #        * MultiheadAttention (self-attention): 512*512*3 + 512*512 = 1.048M
#    #        * MultiheadAttention (encoder-attention): 512*512*3 + 512*512 = 1.048M
#    #        * FFN: 512*2048*2 = 2.097M
#    # Final FC:
#    #   - FC: 512*5000 = 256K (assuming vocab size 5K)
#    # In total:
#    #       ~65 M

@register_model_architecture("avse_avsr_vggtransformer_norm_DCM", "BiModalvggtransformer_avse_avsr_norm_DCM_base")
def BiModalvggtransformer_avsr_norm_DCM_base(args):
    args.audio_input_feat_per_channel = getattr(args, "audio_input_feat_per_channel", 90)
    args.video_input_feat_per_channel = getattr(args, "video_input_feat_per_channel", 512)
    args.audio_vggblock_enc_config = getattr(
        args, "audio_vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
    args.video_vggblock_enc_config = getattr(
        args, "video_vggblock_enc_config", "[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]"
    )
#    args.audio_vggblock_enc_config = 'None'
#    args.video_vggblock_enc_config = 'None'
    args.transformer_enc_config = getattr(
        args, "transformer_enc_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 6"
    )
    args.audio_enc_output_dim = getattr(args, "audio_enc_output_dim", 512)
    args.video_enc_output_dim = getattr(args, "video_enc_output_dim", 512)
    args.tgt_embed_dim = getattr(args, "tgt_embed_dim", 512)
    args.conv_dec_config = getattr(args, "conv_dec_config", "((256, 3, True),) * 4")
#    args.conv_dec_config = 'None'
    args.transformer_dec_config = getattr(
        args, "transformer_dec_config", "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 6"
    )

