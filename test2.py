import numpy as np



def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

dA = np.array([1,2,3,4,5])
dZ = np.array(dA, copy = True)
Z = np.array([-2,-1,0,1,2])
dZ[Z <= 0] = 0

dX = np.asmatrix([[1,2],[3,4]])
#dR = np.sum(dX.T, axis = 1, keepdims=True)
#print(dR)
 #db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
x=np.array([[1,2],[3,4]])
y=[[1,2],[3,4]]
dy=([1,2],[3,4])#tuple
z=[[1,2],[3,4]] #list
#print(y)
# print(np.sum(x, axis = 0))
# print(np.sum(y, axis = 1))
# print(np.sum([[0, 1, 2], [0, 1,5]], axis=0, keepdims= True))

nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
] #dictionary..  input


x = np.array([[1,2,3],
             [4,5,6]])

y = np.array([[2,1,1],
             [2,1,1]])

x = np.array([0.509, 0.663, 0.546, 0.695])
y = np.array([0.491, 0.337, 0.454, 0.305])

cost_y = np.array([.127, -0.084, -0.114, -0.076])
y_z = np.multiply(x, y)
#print(np.multiply(x, y))
    
cost_z = np.multiply(cost_y, y_z)
#print(cost_z)

x_t = np.array([[0,0], [0,1], [1,0], [1,1]])
#print(x_t)

#print(cost_z.dot(x_t))

#print(np.sum(cost_z))


###################

x = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]])

w = np.array([0.1, 0.6, 0.2])
b = np.array([0])
z = w.dot(x) + b
y = np.array([0,1,1,0])
y_hat = np.divide(1, (1+np.exp(-z)))

cost_y_hat = np.divide(y_hat - y, 4)
#print(cost_y_hat)
y_hat_z = np.multiply(y_hat, 1 - y_hat)
#print(y_hat_z)
cost_z = np.multiply(cost_y_hat, y_hat_z)
#print(cost_z)


cost_w = cost_z.dot(x.T)

#print(cost_w)

# ex = np.array([[1,2,3], [-1,-2,-3]])
# ex2 = np.array([[3],[4]])
# print(np.multiply(ex, ex2))

lr = 1 #learning rate
x = np.array([[0,0,1,1], [0,1,0,1]])
y = np.array([0,1,1,0])
w1 = np.array([[0.099, 0.599], [0.199, 0.398], [0.298, 0.697]])
b1 = np.array([[-0.002], [-0.004], [-.007]])

w2 = np.array([[0.076, 0.377, 0.875]])
b2 = np.array([-0.041])
def act_func(z):
    return np.divide(1, (1+np.exp(-z)))

def cost(y, y_hat):
    return np.divide( np.sum(np.square(y - y_hat)),8)

#forward
z1 = w1.dot(x) + b1 #3x2
a1 = act_func(z1) #3x2

z2 = w2.dot(a1) + b2
a2 = act_func(z2)

print(cost(y,a2))


#back
cost_y_hat = np.divide(-(y - a2), 4)

y_hat_z2 = np.multiply(y_hat, (1 - y_hat))

cost_z2 = np.multiply(cost_y_hat, y_hat_z2)

z2_w2 = a1.T
cost_w2 = cost_z2.dot(z2_w2)

z2_a1 = w2.T

cost_a1 = z2_a1.dot(cost_z2)
a1_z1 = np.multiply(a1, (1-a1))
cost_z1 = np.multiply(cost_a1, a1_z1)
z1_w = x.T
cost_w = cost_z1.dot(z1_w)
print(cost_w)
z2_b2 = np.array([1])

cost_b2 = np.sum(np.multiply(cost_z2, z2_b2))

#newb2 -= lr*cost_b2




