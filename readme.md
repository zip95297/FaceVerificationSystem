
# 输入界面

1. 输入图像img1
2. 输入图像img2
3. 模型选择：teacher，student

# 输出界面

1. 判断结果
2. 置信度 |sigmoid(pridiction-threshhode)-0.5|*2
3. 推理时间

# 流程

1. 得到输入图像
2. 将输入图像进行预处理
3. 加载入网络得到embeddings，并记录时间
4. 比对，通过比较模型的treshhold判断是否为同一个人，并得到置信度

# TODO

1. 修改置信度公式

# TIPS

1. 教室模型是在使用了`model = nn.DataParalle()`情况下训练的,加载到CUDA应包含上面的代码。若加载到非CUDA设备，应使用[这个](./models/util_pth.py)进行处理。
