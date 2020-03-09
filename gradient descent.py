# Imports
#%matplotlib notebook

import sys
import numpy as np  # Matrix and vector computation package
import matplotlib
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Fancier plots

# Set seaborn plotting style
sns.set_style('darkgrid')
# Set the seed for reproducability
np.random.seed(seed=13)
#

x = np.random.uniform(0, 1, 20)

def f(x):
    return x*2

noise_variance = 0.2
#print(x)
noise = np.random.randn(x.shape[0]) * noise_variance

#print(noise)
t = f(x) + noise

# Plot the target t versus the input x
plt.figure(figsize=(5, 3))
plt.plot(x, t, 'o', label='$t$')
# Plot the initial line
plt.plot([0, 1], [f(0), f(1)], 'b--', label='$f(x)$')
plt.xlabel('$x$', fontsize=12)
plt.ylabel('$t$', fontsize=12)
plt.axis((0, 1, 0, 2))
plt.title('inputs (x) vs targets (t)')
plt.legend(loc=2)
plt.show()
#

def nn(x, w):
    print(x)
    print(w)
    return x * w

def loss(y, t):
    return np.mean((t - y) ** 2)

ws = np.linspace(0, 4, num = 100)

# x = np.random.uniform(0, 1, 20)
# t = f(x) + noise
#loss_ws = np.vectorize((lambda w: loss(nn(x, w) , t))(ws)) #(1) put ws into w (2) put w into nn(x, w) (3) 
loss_ws = np.vectorize(lambda w: loss(nn(x, w) , t))(ws)
plt.figure(figsize=(5, 3))
plt.plot(ws, loss_ws, 'r--', label='loss')
plt.xlabel('$w$', fontsize=12)
plt.ylabel('$\\xi$', fontsize=12)
plt.title('loss function with respect to $w$')
plt.xlim(0, 4)
plt.legend()
plt.show()



def gradient(w, x, t):
    return 2*x*(nn(x,w) - t)

def delta_w(w_k, x, t, learning_rate):
    return learning_rate*np.mean(gradient(w_k, x, t))


w = np.random.rand()
learning_rate = 0.9

nb_of_iterations = 4
w_loss = [(w, loss(nn(x,w), t))]