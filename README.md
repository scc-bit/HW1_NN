# HW1_NN

## Neural_Network文件夹说明

- **train.py**：参数查找及训练，参数查找包括学习率，隐藏层大小，正则化强度

- **test.py**：测试，导入模型，用经过参数查找后的模型进行测试，输出分类精度

- loaddata.py：加载MNIST数据集的训练集和测试集

- NN.py：定义两层神经网络，包含激活函数、反向传播，loss以及梯度的计算、学习率下降策略、L2正则化、优化器SGD

- Data文件夹：原始MNIST数据集

### 输出文件

- bestmodel.npz：参数查找后验证集acc最高的模型

- loss_and_acc.png：bestmodel训练过程的loss,训练和验证过程的accuracy

- visualize_weights.png：bestmodel的网络参数可视化

- trainrecord.log：训练过程参数查找的记录

## 示例命令 

进入到主文件夹：cd Neural_Network

### 训练

python train.py 

程序将保存参数查找后的模型bestmodel.npz

绘制bestmodel训练过程的loss,训练和验证过程的accuracy情况并输出loss_and_acc.png

绘制bestmodel的网络参数并输出loss_and_acc.png

### 测试

python test.py 

程序将导入模型bestmodel.npz并打印输出测试集上的accuracy
