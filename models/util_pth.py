import torch
import torch.nn as nn
from resnet import ResIRSE
from resnet18 import ResNet18
from collections import OrderedDict

# Teacher train with data parallel

state_dict = torch.load("models/teacher.pth", map_location='cpu')

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

model = ResIRSE(embedding_size=512,drop_ratio=0.5)


model.load_state_dict(new_state_dict)

torch.save(model.state_dict(), "TeacherWithoutDataParalle.pth")


# Student train without data parallel

model = ResNet18()
state_dict = torch.load("models/student.pth", map_location='cpu')
model.load_state_dict(state_dict)
torch.save(model.state_dict(), "StudentWithoutDataParalle.pth")