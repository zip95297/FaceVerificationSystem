from torchsummary import summary as smy
from resnet18 import ResNet18
from resnet import ResIRSE

model = ResIRSE(embedding_size=512,drop_ratio=0.5)
smy(model, (1, 128, 128))