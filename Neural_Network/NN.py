import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, hidden_size)/ np.sqrt(input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)/ np.sqrt(hidden_size)
        self.b2 = np.zeros(output_size)
    
    def forward(self, X): 
        h1 = np.maximum(0, np.dot(X, self.W1) + self.b1)  #relu激活函数
        scores = np.dot(h1, self.W2) + self.b2
        return scores, h1 
    
    def backward(self, X, y, scores, h1, reg): #反向传播
        grads = {}  
        num_train = X.shape[0]  
        dscores = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) #softmax
        dscores[range(num_train),y] -= 1  
        dscores /= num_train 
        dh1 = np.dot(dscores, self.W2.T) 
        dh1[h1 <= 0] = 0
        grads['W2'] = np.dot(h1.T, dscores) + reg * self.W2  #梯度计算
        grads['b2'] = np.sum(dscores, axis=0) 
        grads['W1'] = np.dot(X.T, dh1) + reg * self.W1 
        grads['b1'] = np.sum(dh1, axis=0) 
        return grads  
    
    def train(self, X_train, y_train, X_val, y_val, learning_rate=1e-3, reg=1e-5,batch_size=64,num_epochs = 100):
        num_train = X_train.shape[0]
        iters_per_epoch = num_train // batch_size
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        for epoch in range(num_epochs): 
            loss = 0   
            for i in range(iters_per_epoch):            
                batch_indices = np.random.choice(num_train, batch_size, replace=True)  
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                scores, h1 = self.forward(X_batch)
                loss += self.compute_loss(scores, y_batch, reg)
                grads = self.backward(X_batch, y_batch, scores, h1, reg)
                self.W1 -= learning_rate * grads['W1']
                self.b1 -= learning_rate * grads['b1']
                self.W2 -= learning_rate * grads['W2']
                self.b2 -= learning_rate * grads['b2']
                            
            loss_history.append(loss/iters_per_epoch)
            train_acc = (self.predict(X_train) == y_train).mean()                
            val_acc = (self.predict(X_val) == y_val).mean()                
            train_acc_history.append(train_acc)                
            val_acc_history.append(val_acc)
            
        
        if epoch > 0 and epoch % 5 == 0:
            learning_rate *= 0.99  #学习率下降策略

        return {
            'loss_history': loss_history,'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }
    
    def predict(self, X):
        scores, _ = self.forward(X)
        return np.argmax(scores, axis=1)
    
    def compute_loss(self, scores, y,  reg): #loss计算
        num_train = scores.shape[0]  
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  
        correct_logprobs = -np.log(probs[np.arange(num_train),y]) 
        data_loss = np.sum(correct_logprobs) / num_train
        reg_loss = 0.5 * reg * (np.sum(self.W1 * self.W1) + np.sum(self.W2* self.W2)) #L2正则化
        loss = data_loss + reg_loss
        return loss
    
    
    def get_accuracy(self, X, y):
        predicted_y = self.predict(X)
        accuracy = np.mean(predicted_y == y)
        return accuracy
    
    def plot_loss_accuracy(self, stats):  #可视化loss和acc
        plt.figure (figsize = (16,8))
        ax1=plt.subplot(121)
        ax1.xaxis.get_major_locator().set_params(integer=True)
        plt.plot(stats['loss_history'], label='train loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Train loss')
        plt.legend()
        
        ax2=plt.subplot(122)
        ax2.xaxis.get_major_locator().set_params(integer=True)
        plt.plot(stats['train_acc_history'], label='train acc')
        plt.plot(stats['val_acc_history'], label='val acc')
        plt.title('Classification accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig("loss_and_acc.png")
 
    def visualize_weights(self):   #可视化weights
        plt.figure (figsize = (8,8))
        plt.subplot(1, 2, 1)
        plt.imshow(self.W1)
        plt.title('First layer weights')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.imshow(self.W2)
        plt.title('Second layer weights')
        plt.colorbar()
        plt.savefig("visualize_weights.png")
