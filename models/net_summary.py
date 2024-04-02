from torchsummary import summary as smy
from resnet18 import ResNet18
from resnet import ResIRSE
import torch
from thop import profile

#model = ResNet18()
model = ResIRSE(embedding_size=512,drop_ratio=0.5)
smy(model, input_size=( 1, 128, 128), batch_size=64)

input = torch.randn(1, 1, 128, 128)
twice_flops, params = profile(model, inputs=(input, ), verbose=False)

print(f"FLOPS: {twice_flops/2}, Params: {params}")
if hasattr(model,"flops") :
    print(f"{model.flops((1,128,128))}")