import numpy as np
first_layer_size = 4
train = 10


zero_array = np.array([
    [77, 255, 255, 9],
    [255, 100, 100, 255],
    [255, 100, 150, 24],
    [168, 255, 54, 123]
], dtype=np.uint8)  # Using dtype=np.uint8 for grayscale values between 0-255

zero_array = zero_array / 255
zero_array_flatten = np.concatenate(zero_array, axis=None)


def init_weights ():
    weights = np.random.rand(len(zero_array_flatten), first_layer_size)
    return weights

def init_bias():
    bias = np.random.rand(1, first_layer_size)
    return bias[0]*5

weights = init_weights()
bias = init_bias()
print(weights, bias)

def weighed_sum(weights, bias, flat_array):
        sum = 0
        biascurrent = 0
        for idxW, elemW in enumerate(weights[0]):
            for idx, elem in enumerate(flat_array):
                sum = sum + elem * weights[idx][idxW]
                biascurrent = bias[idxW]
            sum = sigmoid(sum - biascurrent)
            print(sum)
            sum = 0

def find_desc(weights):
    print("not done")
     

for elem in range(train):
    def sigmoid(x):
        return 1/(1+np.exp((-x)))

    weighed_sum(weights, bias, zero_array_flatten)


# To visualize the array as an image:
import matplotlib.pyplot as plt
plt.imshow(zero_array, cmap='gray', vmin = 0, vmax = 1)
plt.show()

#print("Array: ", zero_array_flatten)

