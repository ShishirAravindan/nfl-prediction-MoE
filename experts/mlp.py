import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from etl.transform_mlp import load_mlp_features

def make_onehot(indicies, total=128):
    I = np.eye(total)
    return I[indicies]

def softmax(z):
    m = np.amax(z, axis=-1, keepdims=True)
    return np.exp(z - m) / np.sum(np.exp(z - m), axis=-1, keepdims=True)

def relu(m):
  return np.maximum(m, 0)

def relu_prime(m):
  return np.where(m > 0, 1, 0)

def tanh(m):
  return np.tanh(m)

def tanh_prime(m):
  return 1 - np.tanh(m)**2

def do_forward_pass(model, X):
    model.N = X.shape[0]
    model.X = X

    model.m1 = np.add(np.dot(model.X, model.W1), model.b1) # TODO - the hidden state value (pre-activation)
    model.h1 = relu(model.m1) # TODO - the hidden state value (post ReLU activation)

    model.m2 = np.add(np.dot(model.h1, model.W2), model.b2) # TODO - the hidden state value (pre-activation)
    model.h2 = relu(model.m2) # TODO - the hidden state value (post ReLU activation)

    model.m3 = np.add(np.dot(model.h2, model.W3), model.b3) # TODO - the hidden state value (pre-activation)
    model.h3 = relu(model.m3) # TODO - the hidden state value (post ReLU activation)

    model.z = np.add(np.dot(model.h3, model.W4), model.b4) # TODO - the logit scores (pre-activation)
    model.y = softmax(model.z) # TODO - the class probabilities (post-activation)
    return model.y

def do_backward_pass(model, ts):

    model.z_bar = model.y - ts
    model.W4_bar = np.dot(model.h3.T, model.z_bar) / model.N
    model.b4_bar = np.sum(model.z_bar, axis=0) / model.N

    model.h3_bar = np.dot(model.z_bar, model.W4.T)
    model.m3_bar = model.h3_bar * relu_prime(model.m3)

    model.W3_bar = np.dot(model.h2.T, model.m3_bar) / model.N
    model.b3_bar = np.sum(model.m3_bar, axis=0) / model.N

    model.h2_bar = np.dot(model.m3_bar, model.W3.T)
    model.m2_bar = model.h2_bar * relu_prime(model.m2)

    model.W2_bar = np.dot(model.h1.T, model.m2_bar) / model.N
    model.b2_bar = np.sum(model.m2_bar, axis=0) / model.N

    model.h1_bar = np.dot(model.m2_bar, model.W2.T)
    model.m1_bar = model.h1_bar * relu_prime(model.m1)

    model.W1_bar = np.dot(model.X.T, model.m1_bar) / model.N
    model.b1_bar = np.sum(model.m1_bar, axis=0) / model.N

def train_sgd(model, X_train, t_train,
              alpha=0.1, n_epochs=0, batch_size=100,
              X_valid=None, t_valid=None,
              w_init=None, plot=True):
    # as before, initialize all the weights to zeros
    w = np.zeros(X_train.shape[1])

    train_loss = [] # for the current minibatch, tracked once per iteration
    valid_loss = [] # for the entire validation data set, tracked once per epoch

    # track the number of iterations
    niter = 0

    # we will use these indices to help shuffle X_train
    N = X_train.shape[0] # number of training data points
    indices = list(range(N))

    for e in range(n_epochs):
        random.shuffle(indices) # for creating new minibatches

        for i in range(0, N, batch_size):
            if (i + batch_size) > N:
                # At the very end of an epoch, if there are not enough
                # data points to form an entire batch, then skip this batch
                continue

            #generate minibatch
            X_minibatch = X_train[indices[i:i+batch_size]]
            t_minibatch = t_train[indices[i:i+batch_size]]

            # gradient descent iteration
            model.cleanup()
            model.forward(X_minibatch)
            model.backward(t_minibatch)
            model.update(alpha)

            if plot:
                # Record the current training loss values
                loss = model.loss(t_minibatch.astype(int))
                train_loss.append(loss)
            niter += 1

        # compute validation data metrics, if provided, once per epoch
        if plot and (X_valid is not None) and (t_valid is not None):
            model.cleanup()
            model.forward(X_valid)
            valid_loss.append((niter, model.loss(t_valid.astype(int))))

    if plot:
        plt.title("SGD Training Curve Showing Loss at each Iteration")
        plt.plot(train_loss, label="Training Loss")
        if (X_valid is not None) and (t_valid is not None): # compute validation data metrics, if provided
            plt.plot([iter for (iter, loss) in valid_loss],
                    [loss for (iter, loss) in valid_loss],
                    label="Validation Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        print("Final Training Loss:", train_loss[-1])
        if (X_valid is not None) and (t_valid is not None):
            print("Final Validation Loss:", valid_loss[-1])

class GameContextMLP(object):

    def __init__(self, num_features=128*20, num_hidden=50, num_classes=2, pretrained=False):
      """
      Initialize the weights and biases of this multi-layer MLP.
      """
      # Model architecture
      self.num_features = num_features
      self.num_hidden = num_hidden
      self.num_classes = num_classes

      # Weights and biases for the first hidden layer
      self.W1 = np.zeros([num_features, num_hidden])
      self.b1 = np.zeros([num_hidden])

      # Weights and biases for the second hidden layer
      self.W2 = np.zeros([num_hidden, num_hidden])
      self.b2 = np.zeros([num_hidden])

      # Weights and biases for the third hidden layer
      self.W3 = np.zeros([num_hidden, num_hidden])
      self.b3 = np.zeros([num_hidden])

      # Weights and biases for the output layer
      self.W4 = np.zeros([num_hidden, num_classes])
      self.b4 = np.zeros([num_classes])

      # Initialize weights and biases
      self.initializeParams()

      # Cleanup intermediate variables
      self.cleanup()

      if pretrained:
        self.load_pretrained()

    def load_pretrained(self):
        checkpoint = torch.load(f'../checkpoints/game_context.pth')
        self.load_state_dict(checkpoint)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def initializeParams(self):
        """
        Randomly initialize the weights and biases.
        """
        self.W1 = np.random.normal(0, 2/self.num_features, self.W1.shape)
        self.b1 = np.random.normal(0, 2/self.num_features, self.b1.shape)
        self.W2 = np.random.normal(0, 2/self.num_hidden, self.W2.shape)
        self.b2 = np.random.normal(0, 2/self.num_hidden, self.b2.shape)
        self.W3 = np.random.normal(0, 2/self.num_hidden, self.W3.shape)
        self.b3 = np.random.normal(0, 2/self.num_hidden, self.b3.shape)
        self.W4 = np.random.normal(0, 2/self.num_hidden, self.W4.shape)
        self.b4 = np.random.normal(0, 2/self.num_hidden, self.b4.shape)

    def forward(self, X):
        return do_forward_pass(self, X)

    def backward(self, ts):
        return do_backward_pass(self, ts)

    def loss(self, ts):
        return np.sum(-ts * np.log(self.y)) / ts.shape[0]

    def cleanup(self):
        self.X = None
        self.m1, self.h1 = None, None
        self.m2, self.h2 = None, None
        self.m3, self.h3 = None, None
        self.z, self.y = None, None

        self.z_bar = None
        self.W4_bar, self.b4_bar = None, None
        self.W3_bar, self.b3_bar = None, None
        self.W2_bar, self.b2_bar = None, None
        self.W1_bar, self.b1_bar = None, None

    def update(self, alpha):
        self.W1 -= alpha * self.W1_bar
        self.b1 -= alpha * self.b1_bar
        self.W2 -= alpha * self.W2_bar
        self.b2 -= alpha * self.b2_bar
        self.W3 -= alpha * self.W3_bar
        self.b3 -= alpha * self.b3_bar
        self.W4 -= alpha * self.W4_bar
        self.b4 -= alpha * self.b4_bar

def train_mlp():
    X_train, y_train, X_test, y_test, X_valid, y_valid = load_mlp_features()
    model = GameContextMLP(num_features=X_train.shape[1])
    # Best parameters found using grid search:
    # {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'sgd'}
    train_sgd(model, X_train=X_train, t_train=y_train, X_valid=X_valid, t_valid=y_valid, alpha=0.0001, batch_size=100, n_epochs=30, plot=False)

if __name__ == "__main__":
    train_mlp()