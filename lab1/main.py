""" Deep Learning Practicing: Lab1 - Backpropagation
- Objective: Implement a simple neural network with forward and backward pass using two hidden layers
- Description: 
    - Limitation: "Numpy", "matplotlib" only
    - Visualize: plot predictions, ground truth, learning curve(loss, epoch), accuracy
    - Each layer should contain at least one transformation and one activation function
    - Training n times or until convergence
    - Linear data & XOR data
- Extra: 
    - different optimizers, different activation functions, convolutional layers
"""

"""
hidden_layer (num, inp_dims, out_dims, size)
percptron (?)
Net ()
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# data generater
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
        
    return np.array(inputs), np.array(labels).reshape(21, 1)

# visualization
def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
            
    plt.show()    

def show_loss_curve(costs, optimizer, activation):
    x = list(range(1, len(costs) + 1))
    plt.xlabel('epoch'), plt.ylabel('MSE'), plt.title(f"Training curve with '{optimizer}', '{activation}'")
    plt.plot(x, costs), plt.show()

# layers
def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def tanh(Z):
    A = np.tanh(Z)
    cache = Z
    return A, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def tanh_backward(dA, cache):
    Z = cache
    dZ = (1 - np.tanh(Z)**2) * dA
    return dZ

# cost function
def mean_square_error(Y_pred, Y):
    return np.mean((Y_pred - Y)**2)

def derivative_mse(Y_pred, Y):
    m = len(Y)
    return 2 * np.mean(Y_pred - Y, axis=0, keepdims=True)

# Network
class Network:
    def __init__(self, optimizer='sgd', activation='relu', layer_dims=[2, 16, 16, 1]) -> None:
        # model setting
        self.layer_dims = layer_dims
        self.optimizer = optimizer
        self.activation = activation
        
        self.parameters = dict()
        self.initialize_parameters()
        
        self.reminder = dict()
    
    def initialize_parameters(self):
        for l in range(1, len(self.layer_dims)):
            self.parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1])
            self.parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
    
    def train(self, X, Y, max_epoch=100000, lr=1e-3):
        costs = []
        for epoch in range(max_epoch):
            # forward to get prediction
            Y_pred = self.forward_propagation(X)
            
            # cost function
            cost = mean_square_error(Y_pred.T, y)
            
            # backward to get gradient
            grads = self.backward_propagation(Y_pred, Y)
            
            # optimize
            self.optimize(lr, grads)
            self.reminder = grads.copy()
            
            if (epoch+1) % 5000 == 0:
                print(f'epoch {epoch+1} loss : {cost}')
            
            costs.append(cost)
        return costs

    def predict(self, X, Y):
        correct = 0
        Y_pred = self.forward_propagation(X)
        print(Y_pred)
        for idx, y in enumerate(Y_pred[0]):
            print(f'Iter{idx}\t|\tGround truth: {Y[idx]} |\tprediction: {Y_pred[0][idx]} |')
            if np.int32(Y[idx][0]) == np.int32(np.round(Y_pred[0][idx])):
                correct += 1
        loss = mean_square_error(Y_pred.T, Y)
        acc = correct / len(Y)
        print('loss={:.3f} accuracy={:.3f}%'.format(loss, acc))
        
        return np.round(Y_pred)
    
    def optimize(self, lr, grads, beta=0.8):
        L = len(self.parameters) // 2
        if self.optimizer == 'sgd':
            for l in range(0, L):
                self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - lr * grads["dW" + str(l+1)]
                self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - lr * grads["db" + str(l+1)]
            
        elif self.optimizer == 'momentum':
            for l in range(0, L):
                if len(self.reminder) == 0:
                    Wt = grads["dW" + str(l+1)]
                    bt = grads["db" + str(l+1)]
                else:
                    Wt = beta * self.reminder["dW" + str(l+1)] - lr * grads["dW" + str(l+1)]
                    bt = beta * self.reminder["db" + str(l+1)] - lr * grads["db" + str(l+1)]
                    
                self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - Wt
                self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - bt

    def forward_propagation(self, X):
        def linear_forward(A, W, b):
            Z = np.dot(W,A) + b
            cache = (A, W, b)
            return Z, cache
        
        def linear_activation_forward(A_prev, W, b, activation):
            if activation == "sigmoid":
                Z, linear_cache = linear_forward(A_prev, W, b)
                A, activation_cache = sigmoid(Z)
            elif activation == "relu":
                Z, linear_cache = linear_forward(A_prev, W, b)
                A, activation_cache = relu(Z)
            elif activation == "tanh":
                Z, linear_cache = linear_forward(A_prev, W, b)
                A, activation_cache = tanh(Z)
            elif activation == None:
                Z, linear_cache = linear_forward(A_prev, W, b)
                A, activation_cache = Z, None
                
            cache = (linear_cache, activation_cache)

            return A, cache
        
        self.caches = []
        A = X
        L = len(self.parameters) // 2               

        # [LINEAR -> RELU]*(L-1).
        for l in range(1, L):
            A_prev = A
            A, cache = linear_activation_forward(A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], activation=self.activation)
            self.caches.append(cache)
            
        # LINEAR -> SIGMOID.
        AL, cache = linear_activation_forward(A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], activation="sigmoid")
        self.caches.append(cache)
        
        return AL

    def backward_propagation(self, AL, Y):
        def linear_backward(dZ, cache):
            A_prev, W, b = cache
            m = A_prev.shape[1]

            dW = 1/m * (np.dot(dZ, A_prev.T))
            db = 1/m * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)

            return dA_prev, dW, db
        
        def linear_activation_backward(dA, cache, activation):
            linear_cache, activation_cache = cache

            if activation == "relu":
                dZ = relu_backward(dA, activation_cache)
                dA_prev, dW, db = linear_backward(dZ, linear_cache)
            elif activation == "sigmoid":
                dZ = sigmoid_backward(dA, activation_cache)
                dA_prev, dW, db = linear_backward(dZ, linear_cache)
            elif activation == "tanh":
                dZ = tanh_backward(dA, activation_cache)
                dA_prev, dW, db = linear_backward(dZ, linear_cache)
            elif activation == None:
                dA_prev, dW, db = linear_backward(dA, linear_cache)
            

            return dA_prev, dW, db
        
        grads = dict()
        L = len(self.caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

        # Loss -> Last Layer
        dAL = derivative_mse(AL, Y)
        current_cache = self.caches[-1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

        # lth layer -> (l-1)th layer
        for l in reversed(range(L-1)):
            current_cache = self.caches[l]
            dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, self.activation)
            grads["dA" + str(l + 1)] = dA_prev
            grads["dW" + str(l + 1)] = dW
            grads["db" + str(l + 1)] = db
            
        return grads


if __name__ == '__main__':
    # hyperparameter
    lr = 1e-3
    max_epoch = 50000
    optimizer = 'sgd'
    activation = 'relu'
    layer_dims = [2, 16, 16, 1]
    
    # generate data
    x, y = generate_linear(100)
    
    # model
    model = Network(optimizer=optimizer, activation=activation, layer_dims=layer_dims)
    costs = model.train(x.T, y, max_epoch=max_epoch, lr=lr)
    y_pred = model.predict(x.T, y)
    show_result(x, y, y_pred.T)
    show_loss_curve(costs, optimizer, activation)
    
    # generate data
    x, y = generate_XOR_easy()
    
    # model
    model = Network(optimizer=optimizer, activation=activation, layer_dims=layer_dims)
    costs = model.train(x.T, y, max_epoch=max_epoch, lr=lr)
    y_pred = model.predict(x.T, y)
    show_result(x, y, y_pred.T)
    show_loss_curve(costs, optimizer, activation)