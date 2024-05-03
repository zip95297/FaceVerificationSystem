import torch
import torch.nn as nn
import torch.optim as optim
from .resnet import ResIRSE
from .resnet18 import ResNet18
from .resnet18_with_cfg import ResNet18_with_config
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
            self.model.load_state_dict(torch.load("models/StudentWithoutDataParalle.pth"))
            self.treshold = 0.3187718987464905
        elif self.model_name == "pruned_resnet18":
            self.model= ResNet18_with_config(config=[512, 60, 'M', 64, 64, 64, 63, 128, 128, 53, 128, 128, 256, 256, 8, 247, 222, 497, 377, 5, 420, 510]).to(self.device)
            self.model.load_state_dict(torch.load("models/pruned_model.pth",map_location=self.device))
            self.treshold = 0.2431570739
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
            self.result = "SAME person" if self.cosin_simularity > self.treshold else "NOT SAME person"
            end = time.time()
            self.inference_time = end - start
            
            self.confidence =abs(1/(1+math.e**(-(self.cosin_simularity - self.treshold)*8))-0.5)*2
            self.confidence_precent= self.confidence*100
            return self.result, self.confidence_precent, self.inference_time
        
    def cosin_metric(self,x1, x2):
        # 直接使用tensor计算
        x1=x1.squeeze()
        x2=x2.view(-1)
        return torch.dot(x1, x2) / (torch.norm(x1) * torch.norm(x2))
        # return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def get_result(self):
        return self.result, self.confidence_precent, self.inference_time
    
if __name__ == "__main__":
    # Test test
    img1_pth="/Users/zip95297/Downloads/人脸测试/001.jpg"
    img2_pth="/Users/zip95297/Downloads/人脸测试/AbeVigoda_0001.jpg"

    test = Test("teacher_resnet50", img1_pth, img2_pth)
    test = Test("pruned_resnet18", img1_pth, img2_pth)
    print(test.get_result())