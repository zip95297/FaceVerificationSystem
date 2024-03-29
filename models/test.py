import torch
import torch.nn as nn
import torch.optim as optim
from .resnet import ResIRSE
from .resnet18 import ResNet18
from torchvision import transforms as T
from PIL import Image
import numpy as np
import math
import time
# cuda训练的模型不能直接在apple silicon上推理，需要通过coremltools转换
# pip install -U coremltools
import coremltools as ct

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
        self.inference_time = None
        self.load_model()
        self.load_image()
        self.predict()

    def load_model(self):
        print(f"{self.model_name} model loading on device {self.device}")
        if self.model_name == "teacher_resnet50":
            self.model = ResIRSE(embedding_size=512,drop_ratio=0.5).to(self.device)
            self.treshold = 0.278049
            self.model.load_state_dict(torch.load("models/TeacherWithoutDataParalle.pth"))
        elif self.model_name == "student_resnet18":
            self.model = ResNet18().to(self.device)
            self.model.load_state_dict(torch.load("models/StudentWithoutDataParalle.pth"), map_location=self.device)
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
            self.model.eval()

            # 记录推理时间
            start = time.time()
            output1 = self.model(self.img1)
            output2 = self.model(self.img2)
            self.cosin_simularity = self.cosin_metric(output1, output2)
            self.result = "same person" if self.cosin_simularity > self.treshold else "not same person"
            end = time.time()
            inference_time = end - start

            self.confidence =math.abs(1/(1+math.e**(-(self.cosin_simularity - self.treshold)))-0.5)*2
            self.confidence_precent= self.confidence*100
            return self.result, self.confidence_precent, self.inference_time
        
    def cosin_metric(x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def get_result(self):
        return self.result, self.confidence_precent, self.inference_time
    
if __name__ == "__main__":
    # Test test
    test = Test("teacher_resnet50", "/Users/zip95297/Downloads/001.jpg", "/Users/zip95297/Downloads/AbeVigoda_0001.jpg")
    print(test.get_result())