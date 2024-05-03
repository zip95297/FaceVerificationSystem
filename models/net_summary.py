from torchsummary import summary as smy
from resnet18 import ResNet18
from resnet import ResIRSE
from resnet18_with_cfg import ResNet18_with_config
import torch
from thop import profile

model = ResIRSE(embedding_size=512,drop_ratio=0.5)
model = ResNet18()
model = ResNet18_with_config(config= [512, 60, 'M', 64, 64, 64, 63, 128, 128, 53, 128, 128, 256, 256, 8, 247, 222, 497, 377, 5, 420, 510])

smy(model, input_size=( 1, 128, 128), batch_size=64)

input = torch.randn(1, 1, 128, 128)
twice_flops, params = profile(model, inputs=(input, ), verbose=False)

print(f"FLOPS: {twice_flops/2}, Params: {params}")
if hasattr(model,"flops") :
    print(f"{model.flops((1,128,128))}")