import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy
import pdb
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
        
        self.fc0v = nn.Linear(512,321)
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=4)
        
    def forward(self, video,mag):
        
        video = self.up(video) #B 512 T -> B 512 4T
        video_out = self.fc0v(video.permute(0,2,1)) # [B 512 T] ->[B T 321]
        #video_out,_ = self.rnn_v(video_out)
        video_out = video_out.permute(0,2,1)
        #video_out_weigth = self.sigmoid(video_out_weigth)
        #video_out = (video_out*video_out_weigth).permute(0,2,1)
        
        #video_out = self.video_encoder(video_out.permute(0,2,1)) #[1,1536,4x]
        #video_out = self.fc1v(video_out.permute(0,2,1)).permute(0,2,1)
        video_out = torch.unsqueeze(video_out,1) #[B 1 321 4T]
        
    
        amp_spec = torch.sqrt(mag)
        
        
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
            #print(xs[self.model_length - 1 - i].shape)
            #print(f"p{i}, {p.shape} + x{self.model_length - 1 - i}, {xs[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")
            else:
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
                                (2, 1),
                                (2, 1),
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
                                (2, 1),
                                (2, 1),
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
            self.dec_output_paddings = [(0,0),
                                        (1,0),
                                        (0,0),
                                        (0,0),
                                        (0,0),
                                        (0,0),
                                        (0,0),
                                        (0,0),
                                        (0,0),
                                        (0,0)]
            self.CD_input = [321,
                       321,
                       161,
                       81,
                       41,
                       21,
                       11,
                       6,
                       3,
                       2]
            
            self.CD_input2 = [2,3,
                       6,
                       11,
                       21,
                       41,
                       81,
                       161,
                       321,
                       321]
            
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
