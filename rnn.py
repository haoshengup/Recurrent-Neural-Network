import numpy as np

class NetWork(object):
    def __init__(self, sizes):
        self.sizes = sizes

        # initialize weights
        self.w_xh = np.random.randn(self.sizes[1], self.sizes[0])/np.sqrt(self.sizes[0])
        self.w_hy = np.random.randn(self.sizes[2], self.sizes[1])/np.sqrt(self.sizes[1])
        self.w_hh = np.random.randn(self.sizes[1], self.sizes[1])/np.sqrt(self.sizes[1])
       
        # initialize derivative of weights
        self.w_xh_derivative = np.zeros_like(self.w_xh)
        self.w_hy_derivative = np.zeros_like(self.w_hy)
        self.w_hh_derivative = np.zeros_like(self.w_hh)

        # a list to preserve binary data
        binary_dim = 8
        self.largest_num = 2 ** binary_dim
        binary = np.unpackbits(np.array(list(range(self.largest_num)),dtype=np.uint8)[:,np.newaxis], axis=1)
        self.int2binary = [binary[i,:] for i in range(self.largest_num)]

    def forward(self, x):
        step = x.shape[1]
        layer_h = np.zeros(shape=(step + 1, self.sizes[1], 1))
        layer_out = np.zeros(shape=(step, 1))
        for position in range(step):
            layer_h[position + 1] = self.sigmoid(np.dot(self.w_xh, x[:,position][:,np.newaxis]) + np.dot(self.w_hh, layer_h[position]))
            layer_out[position] = self.sigmoid(np.dot(self.w_hy, layer_h[position + 1]))
        return layer_h, layer_out

    def backward(self, x, out, target):
        step = len(target)

        # calculate error of output
        delta_layer_out = [self.cost_derivative(out[1][i], target[i]) * self.sigmoid_derivative(out[1][i]) for i in range(step)]

        # calculate error of hidden layer
        delta_layer_h = [np.zeros(shape=(self.sizes[1], 1))] * (step + 1)
        for i in reversed(list(range(step))):
            delta_layer_h[i] = (np.dot(self.w_hy.T, delta_layer_out[i])[:,np.newaxis] + np.dot(self.w_hh.T, delta_layer_h[i + 1])) \
                    * self.sigmoid_derivative(out[0][i + 1])

        # calculate derivative of weights
        self.w_hy_derivative = sum([np.dot(delta_layer_out[i], out[0][i + 1].T) for i in range(step)])
        self.w_hh_derivative = sum([np.dot(delta_layer_h[i], out[0][i].T) for i in range(step + 1)])
        self.w_xh_derivative = sum([np.dot(delta_layer_h[i], x[:,i][:,np.newaxis].T) for i in range(step)])

    def train(self, beta, epoches):
        for j in range(epoches):
            a_int = np.random.randint(self.largest_num/2)
            a = self.int2binary[a_int]

            b_int = np.random.randint(self.largest_num/2)
            b = self.int2binary[b_int]

            c_int = a_int + b_int
            c = self.int2binary[c_int]

            d_int = 0

            x = np.zeros(shape=(2, len(a)))
            x[0] = a[::-1]
            x[1] = b[::-1]

            out = self.forward(x)
            self.backward(x, out, c[::-1])
            self.w_xh -= beta * self.w_xh_derivative
            self.w_hh -= beta * self.w_hh_derivative
            self.w_hy -= beta * self.w_hy_derivative
           
            cost = self.cost(out[-1], c[:,np.newaxis])/len(a)

            d = np.round(out[-1]).reshape(8).tolist()
            for index, num in enumerate(d):
                d_int += num * 2**index
            d_int = int(d_int)
            if j%1000 == 0:
                print('Epoch ' + str(j) + ':   ' + str(a_int) + " + " + str(b_int) + " = " + str(d_int) + '\n')
            
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def cost_derivative(self, out, target):
        return (out - target)

    def cost(self, out, target):
        return  sum(0.5 * sum((out - target) ** 2))

