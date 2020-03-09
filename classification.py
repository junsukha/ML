# Imports
#%matplotlib notebook

import sys
import numpy as np  # Matrix and vector computation package
import matplotlib
import matplotlib.pyplot as plt  # Plotting library
from matplotlib import cm # Colormaps
from matplotlib.colors import colorConverter, ListedColormap
import seaborn as sns  # Fancier plots

# Set seaborn plotting style
sns.set_style('darkgrid')
# Set the seed for reproducability
np.random.seed(seed=1)
#


nb_of_samples_per_class = 20
red_mean = (-1., 0.)  # The mean of the red class
blue_mean = (1., 0.)  # The mean of the blue class
# Generate samples from both classes
x_red = np.random.randn(nb_of_samples_per_class, 2) + red_mean
x_blue = np.random.randn(nb_of_samples_per_class, 2)  + blue_mean
#print(x_red)


X = np.vstack((x_red, x_blue))#x_red 하고 x_blue 하고 

t = np.vstack((np.zeros((nb_of_samples_per_class,1)), 
               np.ones((nb_of_samples_per_class,1))))


plt.figure(figsize=(6, 4))
plt.plot(x_red[:,0], x_red[:,1], 'r*', label='class: red star')
plt.plot(x_blue[:,0], x_blue[:,1], 'bo', label='class: blue circle')
plt.legend(loc=2)
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.axis([-3, 4, -4, 4])
plt.title('red star vs. blue circle classes in the input space')
plt.show()
#


def logistic(z):
    return 1. / (1 + np.exp(-z))

def nn(x,w):
    return logistic(x.dot(w.T))

def nn_predict(x, w):
    return np.around(nn(x,w))

def loss(y,t):
    return -np.mean(np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-y)))

nb_of_ws = 100

wsa = np.linspace(-5,5, nb_of_ws)
wsb = np.linspace(-5,5, nb_of_ws)

ws_x, ws_y = np.meshgrid(wsa, wsb)


loss_ws = np.zeros((nb_of_ws,nb_of_ws))
#print(np.asmatrix([ws_x[0,0], ws_y[0,0]]))

for i in range(nb_of_ws):
    for j in range(nb_of_ws):
        loss_ws[i,j] = loss(nn(X, np.asmatrix([ws_x[i,j], ws_y[i,j]])), t)#X = 100 *2 matrix, np.asmatrix = 1*2 matrix
        #nn(x, w) = logistic(x.dot(w.T))..   so   100*2 dot 2*1

# Plot the loss function surface
plt.figure(figsize=(6, 4))
plt.contourf(ws_x, ws_y, loss_ws, 20, cmap=cm.viridis)
cbar = plt.colorbar()
cbar.ax.set_ylabel('$\\xi$', fontsize=12)
plt.xlabel('$w_1$', fontsize=12)
plt.ylabel('$w_2$', fontsize=12)
plt.title('Loss function surface')
plt.grid()
plt.show()
#

def gradient(w, x, t):
    return (nn(x, w) - t).T * x

def delta_w(w_k, x, t, learning_rate):
    return learning_rate*gradient(w_k, x, t)

w = np.asmatrix([-4,-2])

learning_rate = 0.05

nb_of_iterations  = 10
w_iter = [w]


for i in range (nb_of_iterations):
    dw = delta_w(w, X, t, learning_rate)
    w = w - dw
    w_iter.append(w)


plt.figure(figsize = (6,4))
plt.contourf(ws_x, ws_y, loss_ws, 20, alpha=0.9, cmap=cm.viridis)
cbar = plt.colorbar()
cbar.ax.set_ylabel('loss')

for i in range(1,4):
    
    w1 = w_iter[i-1]
    w2 = w_iter[i]
    print(w1, w2)
    plt.plot(w1[0,0], w1[0,1], 'o')  # Plot the weight-loss value
    plt.plot([w1[0,0], w2[0,0]], [w1[0,1], w2[0,1]], 'k-')
    plt.text(w1[0,0]-0.2, w1[0,1]+0.4, f'$w({i-1})$', color='k')
# Plot the last weight
w1 = w_iter[3]  
plt.plot(w1[0,0], w1[0,1], 'ko')
plt.text(w1[0,0]-0.2, w1[0,1]+0.4, f'$w({i})$', color='k') 
# Show figure
plt.xlabel('$w_1$', fontsize=12)
plt.ylabel('$w_2$', fontsize=12)
plt.title('Gradient descent updates on loss surface')
plt.show()
#