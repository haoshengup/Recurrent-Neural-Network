import rnn
net = rnn.NetWork([2, 16, 1])
net.train(0.9, 10000)