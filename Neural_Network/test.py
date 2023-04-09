from NN import *
from loaddata import *

model = np.load("bestmodel.npz") #导入训练好模型
modelnet = TwoLayerNeuralNetwork(model['input_size'], model['hidden_size'], model['output_size'])
modelnet.W1,modelnet.b1,modelnet.W2,modelnet.b2  = model["W1"],model["b1"],model["W2"],model["b2"]
_, _, X_test, y_test = load_data()
print('Test accuracy: ', modelnet.get_accuracy(X_test, y_test)) 