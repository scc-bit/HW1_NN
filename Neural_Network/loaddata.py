import gzip  
import numpy as np

def load_data(): 
    with gzip.open('Data/train-images-idx3-ubyte.gz', 'rb') as f:
        X_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
    with gzip.open('Data/train-labels-idx1-ubyte.gz', 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open('Data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        X_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
    with gzip.open('Data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)
    return X_train, y_train, X_test, y_test
