import os
from PyQt5.QtWidgets import  QComboBox, QSlider,QHBoxLayout, QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QWidget, QTextEdit, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from functools import partial
from .test import Test
# import test

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My App")
        self.resize(480, 270)
        self.show()
        
        self.img_path1 = None
        self.img_path2 = None

        self.model = None

        self.result = None
        self.confidence_percent = None
        self.inferrence_time = None

        layout = QVBoxLayout()

        self.show_img_path = QLabel("input img first please")
        self.show_img_path.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.show_img_path)

        # 图像输入部分
        layout_img = QHBoxLayout()

        # img1预览与输入
        layout_img1 = QVBoxLayout()
        self.img_label1 = QLabel("img1 expected to be here!")

        self.img_label1.setAlignment(Qt.AlignCenter)
        self.img_label1.setFixedHeight(200)

        layout_img1.addWidget(self.img_label1)
        self.button1 = QPushButton("select img 1")
        self.button1.clicked.connect(partial(self.load_image, self.img_label1))

        layout_img1.addWidget(self.button1)
        layout_img.addLayout(layout_img1)

        # img2预览与输入
        layout_img2 = QVBoxLayout()
        self.img_label2 = QLabel("img2 expected to be here!")

        self.img_label2.setAlignment(Qt.AlignCenter)
        self.img_label2.setFixedHeight(200)

        layout_img2.addWidget(self.img_label2)
        self.button2 = QPushButton("select img 2")
        self.button2.clicked.connect(partial(self.load_image, self.img_label2))

        layout_img2.addWidget(self.button2)
        layout_img.addLayout(layout_img2)

        layout.addLayout(layout_img)
        

        # 允许拖入上传
        self.setAcceptDrops(True)

        # 选择模型
        layout_model_choose = QHBoxLayout()
        layout_model_choose.addWidget(QLabel("Please choose a model:"))
        
        self.combo_box = QComboBox()
        self.combo_box.setFixedWidth(280)
        self.combo_box.addItems(["teacher_resnet50", "student_resnet18"])
        self.combo_box.currentIndexChanged.connect(self.combo_change)
        layout_model_choose.addWidget(self.combo_box)

        layout.addLayout(layout_model_choose)


        self.button1 = QPushButton("TEST")
        self.button1.clicked.connect(self.test_model)
        layout.addWidget(self.button1)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def combo_change(self, index):
        self.model = self.combo_box.currentText()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        # 获取拖入的文件路径
        file_path = event.mimeData().urls()[0].toLocalFile()
        if not os.path.exists(file_path):
            self.show_img_path.setText(f"check the file path!")
            return
        if not file_path.endswith(('.png', '.jpg')):
            self.show_img_path.setText(f"only support png/jpg!")
            return
        # 判断拖放发生的位置
        pos = event.pos()
        if self.img_label1.geometry().contains(pos):
            pixmap = QPixmap(file_path)
            self.img_label1.setPixmap(pixmap.scaled(200, 200, aspectRatioMode=True))
            self.img_path1 = file_path
            self.show_img_path.setText(f"img1 upload successfully!")
        elif self.img_label2.geometry().contains(pos):
            pixmap = QPixmap(file_path)
            self.img_label2.setPixmap(pixmap.scaled(200, 200, aspectRatioMode=True))
            self.show_img_path.setText(f"img2 upload successfully!")
            self.img_path2 = file_path
        else:
            print("drop at pointed place!")

        if self.img_path1 and self.img_path2:
            self.show_img_path.setText(f"both img uploaded successfully!")

    def test_model(self):
        test=Test(model_name=self.model, img_path1=self.img_path1, img_path2=self.img_path2)
        result, confidence_percent, inferrence_time=test.get_result()
        self.result = result
        self.confidence_percent = confidence_percent
        self.inferrence_time = inferrence_time
        # return result, confidence_percent, inferrence_time
    
    def load_image(self,img_label):
        # 弹出文件对话框选择图像文件
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "图像文件 (*.png *.jpg *.bmp)")
        if file_name:
            # 使用QPixmap加载图像文件并设置到QLabel中显示
            pixmap = QPixmap(file_name)
            img_label.setPixmap(pixmap.scaled(200, 200, aspectRatioMode=True))

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()