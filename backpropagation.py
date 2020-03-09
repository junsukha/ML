# Imports
#%matplotlib notebook

import sys
import numpy as np  # Matrix and vector computation package
import matplotlib
import matplotlib.pyplot as plt  # Plotting library
from matplotlib import cm # Colormaps
from matplotlib.colors import colorConverter, ListedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # 3D plots
import seaborn as sns  # Fancier plots

# Set seaborn plotting style
sns.set_style('darkgrid')
# Set the seed for reproducability
np.random.seed(seed=4)
#

nb_of_samples_per_class = 20
blue_mean = 0
red_left_mean = -2
red_right_mean = 2

std_dev = 0.5

xs_blue = np.random.randn(nb_of_samples_per_class, 1) * std_dev + blue_mean

xs_red = np.vstack((np.random.randn(nb_of_samples_per_class//2, 1) * std_dev + red_left_mean, np.random.randn(nb_of_samples_per_class//2, 1) * std_dev + red_right_mean))

x = np.vstack((xs_blue, xs_red))
t = np.vstack((np.ones((xs_blue.shape[0],1)),
                np.zeros((xs_red.shape[0], 1))))

fig = plt.figure(figsize = (7,1))
plt.xlim(-3,3)
plt.ylim(-1,1)

plt.plot(xs_blue, np.zeros_like(xs_blue), 'bo', alpha = 0.75)

plt.plot(xs_red, np.zeros_like(xs_red), 'r*', alpha = 0.75)

plt.gca().axes.get_yaxis().set_visible(False)
plt.title('Input samples from the blue circles and red star classes')
plt.xlabel('$x$', fontsize=12)
plt.ylabel('$y$', fontsize=12)
fig.subplots_adjust(bottom=0.4, top=0.75)
plt.show()


def rbf(zh):
    return np.exp(-zh**2)

zhs = np.linspace(-5,5,100)
fig = plt.figure(figsize=(5,3))
plt.plot(zhs, rbf(zhs), label = '$e^{-z_h^2}$')
plt.xlabel('$z$', fontsize=12)
plt.title('RBF function')
plt.legend()
plt.xlim(-5, 5)
fig.subplots_adjust(bottom=0.2)
plt.show()

def logistic(zo):
    return 1. /(1. + np.exp(-zo))

def hidden_activation(x, wh):
    return rbf(x * wh)

def output_activation(h, bo):
    return logistic(h + bo)

def nn(x, wh, bo):
    return output_activation(hidden_activation(x, wh), bo)

def nn_predict(w, wh, bo):
    return np.around(nn(w, wh, bo))


def loss(y, t):
    return -np.mean(t * np.log(y) + ((1-t) * np.log(1-y)))

def loss_for_param(x, wh, bo, t):
    return loss(nn(x, wh, bo), t)


grid_size = 200
wsh = np.linspace(-3,3, num = grid_size)
bso = np.linspace(-2.5, 1.5, num = grid_size)
params_x, params_y = np.meshgrid(wsh, bso)

loss_grid = np.zeros((grid_size, grid_size))
print(loss_grid)
for i in range(grid_size):
    for j in range(grid_size):
        loss_grid[i,j] = loss()
