import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet import ResIRSE
from models.resnet18 import ResNet18
from torchvision import transforms as T
from PIL import Image
import numpy as np
import math
import os

class Test():
    def __init__(self,model_name, img_path1, img_path2):
        self.model_name = model_name
        self.img_path1 = img_path1
        self.img_path2 = img_path2
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.treshold = None
        self.confidence = None
        self.result = None
        self.load_model()
        self.load_image()
        self.predict()

    def load_model(self):
        if self.model_name == "teacher_resnet50":
            self.model = ResIRSE()
            self.treshold = 0.278049
            self.model.load_state_dict(torch.load("models/teacher.pth"))
        elif self.model_name == "student_resnet18":
            self.model = ResNet18()
            self.model.load_state_dict(torch.load("models/student.pth"))
            self.treshold = 0.3187718987464905
        else:
            print("model not found")
            return
        self.model.eval()

    def load_image(self):
        self.img1 = Image.open(self.img_path1)
        self.img2 = Image.open(self.img_path2)
        self.transform = T.Compose([
        T.Grayscale(),
        T.Resize((128,128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
        # unsqueeze(0)增加一个维度，变成[1,1,128,128]
        self.img1 = self.transform(self.img1).unsqueeze(0)
        self.img2 = self.transform(self.img2).unsqueeze(0)

    def predict(self):
        with torch.no_grad():
            self.img1 = self.img1.to(self.device)
            self.img2 = self.img2.to(self.device)
            self.model.to(self.device)
            self.model.eval()
            output1 = self.model(self.img1)
            output2 = self.model(self.img2)
            self.cosin_simularity = self.cosin_metric(output1, output2)
            self.confidence =math.abs(1/(1+math.e**(-(self.cosin_simularity - self.treshold)))-0.5)*2
            self.result = "same person" if self.cosin_simularity > self.treshold else "not same person"
            self.confidence_precent= self.confidence*100
            return self.result, self.confidence_precent
        
    def cosin_metric(x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
