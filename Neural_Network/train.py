import numpy as np
from sklearn.model_selection import train_test_split
from NN import *
from loaddata import *

def find_best_params(X_train, y_train, X_val, y_val):  #参数查找，学习率、隐藏层大小、正则化强度
    input_size = X_train.shape[1]
    output_size = 10
    best_val_acc = 0
    best_net = None
    best_state = None
    hidden_sizes = [100,200,300,400,500]
    learning_rates = [1e-1,5e-1,1e-2,5e-2,1e-3,5e-3]
    reg_strengths = [1e-3, 5e-3, 1e-4,5e-4,1e-5,5e-5]
    
    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            for reg in reg_strengths:
                net = TwoLayerNeuralNetwork(input_size, hidden_size, output_size)
                stats = net.train(X_train, y_train, X_val, y_val, learning_rate=learning_rate, reg=reg)
                val_acc = net.get_accuracy(X_val, y_val)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_net = net
                    best_state = stats
                print('hidden_size: %d, learning_rate: %e, reg: %e, validation accuracy: %f' % (hidden_size, learning_rate, reg, val_acc))
    
    return best_net,best_state

if __name__ == "__main__":
    
    X_train, y_train,_, _ = load_data() #加载数据集
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, 
                                                                              random_state=2023, shuffle=True) #划分训练、验证集

    best_net,stats = find_best_params(X_train_split, y_train_split,X_val_split,y_val_split) # 参数查找
    
    best_net.plot_loss_accuracy(stats)
    best_net.visualize_weights()
    
    np.savez("bestmodel.npz",input_size=best_net.input_size, hidden_size=best_net.hidden_size, output_size=best_net.output_size,
              W1=best_net.W1, b1=best_net.b1, W2=best_net.W2, b2=best_net.b2) #保存模型