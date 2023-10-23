from numpy import *
import numpy as np
first_layer_size = 2
output_size = 1

def generate_weights():
    weightsInput = np.random.rand(3, first_layer_size)
    weightsH = np.random.rand(3, output_size)
    return weightsInput, weightsH

def generate_fake_weights():
    weightsInput = [[-0.07, 0.94], [0.22, 0.46], [-0.46, 0.10]]
    weightsH = [[-0.22], [0.58], [0.78]]
    return weightsInput, weightsH


def generate_weighed_sum(input, weights):
    sum = 0
    out = []
    sums = []
    for idxW, elemW in enumerate(weights[0]):
        for idx, elem in enumerate(input):
            sum = sum + elem * weights[idx][idxW]
        out.append(sigmoid(sum))
        sums.append(sum)
        sum = 0
    return out, sums

def sigmoid(x):
        return 1/(1+np.exp((-x)))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

valuesInput = [1, 0, 1]

h1 = generate_weighed_sum(valuesInput, generate_fake_weights()[0])[0]
print(h1)
h1.append(1)
o1 = generate_weighed_sum(h1, generate_fake_weights()[1])
print(o1)

error = o1[0][0] - 1
nodeDelta = -error*sigmoid_prime(o1[1][0]) 
print(nodeDelta)


 