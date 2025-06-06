import numpy as np

# Base node class for computation graph
class Node:
    def __init__(self):
        self.value = None
        self.grad = 0
        self.parents = []
        self.children = []

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

# Input node
class Input(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self):
        pass

    def backward(self):
        pass

# Linear layer: Wx + b
class Linear(Node):
    def __init__(self, W, b, x):
        super().__init__()
        self.W = W
        self.b = b
        self.x = x
        self.parents = [x]

    def forward(self):
        self.value = np.dot(self.W, self.x.value) + self.b

    def backward(self):
        self.x.grad += np.dot(self.W.T, self.grad)
        self.W -= learning_rate * np.dot(self.grad, self.x.value.T)
        self.b -= learning_rate * self.grad

# Sigmoid activation
class Sigmoid(Node):
    def __init__(self, x):
        super().__init__()
        self.x = x
        self.parents = [x]

    def forward(self):
        self.value = 1 / (1 + np.exp(-self.x.value))

    def backward(self):
        self.x.grad += self.grad * self.value * (1 - self.value)

# Cross entropy loss
def cross_entropy_loss(predicted, target):
    epsilon = 1e-12
    predicted = np.clip(predicted, epsilon, 1. - epsilon)
    return -np.sum(target * np.log(predicted))

def cross_entropy_grad(predicted, target):
    return predicted - target

# Run forward pass
def forward_pass(nodes):
    for node in nodes:
        node.forward()

# Run backward pass
def backward_pass(nodes):
    for node in reversed(nodes):
        node.backward()

# Train on one example
def train_on_example(x_data, y_data, W, b):
    x = Input(x_data)
    linear = Linear(W, b, x)
    sigmoid = Sigmoid(linear)
    nodes = [x, linear, sigmoid]

    forward_pass(nodes)
    loss = cross_entropy_loss(sigmoid.value, y_data)

    sigmoid.grad = cross_entropy_grad(sigmoid.value, y_data)
    backward_pass(nodes)

    return loss

# Train entire network
def train_network(X, Y, epochs=100):
    global learning_rate
    input_size = X.shape[1]
    output_size = Y.shape[1]

    W = np.random.randn(output_size, input_size) * 0.01
    b = np.zeros((output_size, 1))
    learning_rate = 0.1

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            x_data = X[i].reshape(-1, 1)
            y_data = Y[i].reshape(-1, 1)
            loss = train_on_example(x_data, y_data, W, b)
            total_loss += loss
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(X):.4f}")

# Example: XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Start training
train_network(X, Y, epochs=1000)
