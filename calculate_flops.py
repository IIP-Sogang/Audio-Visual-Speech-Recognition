from thop import profile
# from rnn_arch_3 import UNet
from cochleanet_11M import Cochleanet
import torch
import pdb
from thop import clever_format
import torch.nn as nn
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# net = UNet()
# input1 = torch.randn(1,512,32)
# input2 = torch.randn(1,321,128)
# input3 = torch.randn(1,321,128)

# print(count_parameters(net))
# flops_full, params_full = profile(net, inputs=(input1,input2,input3))
# print(flops_full, params_full)
# macs, params = clever_format([flops_full, params_full], "%.3f")
# print(macs, params)
# pdb.set_trace()

net = Cochleanet()
input1 = torch.randn(1,512,32)
input2 = torch.randn(1,321,128)


print(count_parameters(net))
flops_full, params_full = profile(net, inputs=(input1,input2))
print(flops_full, params_full)
macs, params = clever_format([flops_full, params_full], "%.3f")
print(macs, params)
pdb.set_trace()

# net = UNet()
# input1 = torch.randn(1,321,128)
# input2 = torch.randn(1,321,128)


# print(count_parameters(net))
# flops_full, params_full = profile(net, inputs=(input1,input2))
# print(flops_full, params_full)
# macs, params = clever_format([flops_full, params_full], "%.3f")
# print(macs, params)
# pdb.set_trace()

# net = UNet()
# input1 = torch.randn(1,1,321,128)


# print(count_parameters(net))
# flops_full, params_full = profile(net, inputs=input1)
# print(flops_full, params_full)
# macs, params = clever_format([flops_full, params_full], "%.3f")
# print(macs, params)
# pdb.set_trace()

input1 = torch.randn(1,152,90).float()
input2 = torch.Tensor([[152]]).long()
input3 = torch.randn(1,32,512).float()
input4 = torch.Tensor([[32]]).long()
input5 = torch.Tensor([[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]]).long()

flops_full, params_full = profile(model, inputs=(input1,input2,input3,input4,input5))
print(flops_full, params_full)
macs, params = clever_format([flops_full, params_full], "%.3f")
print(macs, params)