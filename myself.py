import numpy as np

class NeuralNetwork(object):
    def __init__(self, layers = [2,10,1], activations=['sigmoid', 'sigmoid'], seed=99):
        np.random.seed(seed)
        assert(len(layers) == len(activations)+1) #input layer + two layers(activations)
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i+1], layers[i]))
            self.biases.append(np.random.randn(layers[i+1],1))

    def feedforward(self, x):
        a = np.copy(x)
        z_s = []
        a_s = [a]
        for i in range(len(self.weights)):
            activation_function = self.getActivationFunction(self.activations[i])
            z_s.append(self.weights[i].dot(a) + self.biases[i])
            a = activation_function(z_s[-1])
            a_s.append(a)
        return (z_s, a_s)
    

    
    def getActivationFunction(self, name):
        if(name == 'sigmoid'):
            return lambda x : np.exp(x) / (1+np.exp(x))
        elif(name == 'linear'):
            return lambda x: x
        elif(name == 'relu'):
            def relu(x):
                y = np.copy(x)
                y[y<0] = 0
                return y
            return relu
        else:
            return lambda x: x

    def backpropagation(self,y,z_s,a_s):
        dw = []
        db = []
        deltas = [None]*len(self.weights)
        deltas[-1] = y - a_s[-1]

        for i in reversed(range(len(deltas)-1)):
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.getDerivativeActivationFunction(self.activation_function[i])(z_s[i]))

            batch_size = y.shape[1]
            db = [d.dot()]

np.random.seed(99)
x=np.array([1,2,3])
i=0
batch_size = 2
x_batch = x[i:i+batch_size]
#print(x_batch)

X = 2*np.pi*np.random.rand(10).reshape(1,-1)

print(X.shape[1]) #이땐 column 갯수
len(X) #이땐 그냥 1



Y = np.sin(X)

z=np.array([1,2,3])
#print(z.shape)
#print(len(Y))

a=np.zeros((3,1))
print(a)
a_s=[np.array([a])]
#print(a_s)



a = np.array([[1,2,3],[1,1,1]])
print(np.sum(a, axis =1, keepdims=True))
b = np.array([4,5,6])


print(np.sin(0))

z = np.random.rand(10)



##################################
"""
import numpy as np

class NeuralNetwork(object):
    def __init__(self, layers = [2 , 10, 1], activations=['sigmoid', 'sigmoid']):
        assert(len(layers) == len(activations)+1)
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i+1], layers[i]))
            print('original self weight', self.weights)
            self.biases.append(np.random.randn(layers[i+1], 1))

    def feedforward(self, x):# x= [1,2,3,4] list type
        # return the feedforward value for x
        a = np.copy(x)##-----------------
        z_s = []#list
        a_s = [a]##-----------------
        #print('a_s =', a_s)
        #print(self.weights)
        for i in range(len(self.weights)):
            activation_function = self.getActivationFunction(self.activations[i])
            z_s.append(self.weights[i].dot(a) + self.biases[i])#여기서 a는 항상 앞 layer. 따라서 항상 아귀가 맞는다.
            a = activation_function(z_s[-1])
            a_s.append(a)
        #print('a_s', a_s)
        return (z_s, a_s)
    def backpropagation(self,y, z_s, a_s):
        dw = []  # dC/dW
        db = []  # dC/dB
        deltas = [None] * len(self.weights)  # delta = dC/dZ  known as error for each layer #just initialization
        #print(deltas)
        # insert the last layer error
        #deltas[-1] = ((y-a_s[-1])*(self.getDerivitiveActivationFunction(self.activations[-1]))(z_s[-1]))
        deltas[-1] = (a_s[-1] - y)

        # Perform BackPropagation
        for i in reversed(range(len(deltas)-1)):#len(deltas)-1 인 이유는 맨 마지막 layer 은 이미 input 햇기 때문
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.getDerivitiveActivationFunction(self.activations[i])(z_s[i]))        
        #a= [print(d.shape) for d in deltas]
            #print(deltas[i])
            #print('y', y)
        #print('deltas', deltas)
        batch_size = y.shape[1] #number of columns##-----------------
           # print('a_s', a_s)
        db = [d.dot(np.ones((batch_size,1)))/(2.0*float(batch_size)) for d in deltas]
            #inside a_s, there's original input, 1st layer(excluding input layer) a, 2nd layer a
        dw = [d.dot(a_s[i].T)/(2.0*float(batch_size)) for i,d in enumerate(deltas)]
        #print('dw=', dw)
        # return the derivitives respect to weight matrix and biases

        # for i in range(len(a_s)):
        #     print(a_s[i])        
        return dw, db

    def train(self, x, y, batch_size=10, epochs=100, lr = 0.01):
        # update weights and biases based on the output
        
        for e in range(epochs): 
            i=0
            #print('check_x=',x)
            #print('check_y=',y)
            #print('y =', y.shape[1])
            while(i < y.shape[1]):
                #print(i)
                x_batch = x[0][i:i+batch_size].reshape(1,-1)
                #print('x_batch=',x_batch)
                y_batch = y[0][i:i+batch_size].reshape(1,-1)
                i = i+batch_size
                z_s, a_s = self.feedforward(x_batch)
                #print(a_s)
                dw, db = self.backpropagation(y_batch, z_s, a_s)
                self.weights = [w-lr*dweight for w,dweight in  zip(self.weights, dw)]
                #print('dw =', dw)
                #print('self weight =', self.weights)
                self.biases = [w-lr*dbias for w,dbias in  zip(self.biases, db)]
                #print('check',a_s)
                #print(y_batch)
                #print('a_s', a_s[-1])
                
                loss = np.divide(1,2*batch_size*np.sum(np.square(a_s[-1]-y_batch)))
                print('loss', loss)
                #print("loss = {}".format(np.linalg.norm(a_s[-1]-y_batch)))
                

    def getActivationFunction(self, name):
        if(name == 'sigmoid'):
            return lambda x : np.exp(x)/(1+np.exp(x))
        elif(name == 'linear'):
            return lambda x : x
        elif(name == 'relu'):
            def relu(x):
                y = np.copy(x)
                y[y<0] = 0
                return y
            return relu
        else:
            print('Unknown activation function. linear is used')
            return lambda x: x
        
    
    def getDerivitiveActivationFunction(self, name):
        if(name == 'sigmoid'):
            sig = lambda x : np.exp(x)/(1+np.exp(x))
            return lambda x :sig(x)*(1-sig(x)) 
        elif(name == 'linear'):
            return lambda x: 1
        elif(name == 'relu'):
            def relu_diff(x):
                y = np.copy(x)
                y[y>=0] = 1
                y[y<0] = 0
                return y
            return relu_diff
        else:
            print('Unknown activation function. linear is used')
            return lambda x: 1


if __name__=='__main__':
    import matplotlib.pyplot as plt
    np.random.seed(99)
    nn = NeuralNetwork([1, 100, 1],activations=['sigmoid', 'sigmoid'])#[1, 100, 1]
    X = 2*np.pi*np.random.rand(1000).reshape(1, -1)#1000
    #print('X =', len(X))
    y = np.sin(X)
    
    nn.train(X, y, epochs=1000, batch_size=1000, lr = .1)#epoc 10000  batch 64
    _, a_s = nn.feedforward(X)
    #print(y, X)
    #print('final a_s', a_s)
    plt.scatter(X.flatten(), y.flatten())
    plt.scatter(X.flatten(), a_s[-1].flatten())
    plt.show()

    """