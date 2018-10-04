# MNIST手写字体识别
## 1 开发环境
- Ubuntu
- tensorflow
- python3
- virtualenv
- MNIST数据集
## 2 神经网络
### 2.1前向神经网络
mnist_inference.py
用于训练和测试,无需关心具体的神经结构.
使用python/python3 mnist_inference.py
### 2.2 训练神经网络
mnist_train.py
使用python/python3 mnist_train.py,生成并保存训练模型.
### 2.3 验证测试模型
mnist_eval.py
使用python/python3 mnist_eval.py
## 3 LeNet卷积神经网络
### 3.1 卷积前向传播
mnistcnn_inference.py
用于训练，无需关心内部结构，指定参数即可。
### 3.2 训练神经网络
mnistcnn_train.py
卷积神经网络的输入层是三维矩阵，输入调整为batch、image_size、image_size和num_channels(图像深度)。
python3 mnistcnn_train.py



