import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loss import loss, mse_grad
def load_data():
    data = pd.read_csv("data/clean_weather.csv", index_col=0)
    data = data.ffill() # fill missing values
    return data

if __name__ == '__main__':
    data = load_data()

    np.random.seed(0)
    i_weights = np.random.rand(1, 5)/5 - 0.1
    h_weights = np.random.rand(5, 5)/5 - 0.1
    h_bias = np.random.rand(1, 5)/5 - 0.1

    o_weights = np.random.rand(5, 1) * 50
    o_bias = np.random.rand(1, 1)

    outputs = np.zeros(3)
    hiddens = np.zeros((3, 5))
    prev_hidden = None
    sequence = data["tmax"].tail(3).to_numpy()


    for i in range(outputs.shape[0]):
        x = sequence[i].reshape(1, 1)
        xi = x @ i_weights

        if prev_hidden is None:
            xh = xi
        else:
            xh = xi + prev_hidden @ h_weights + h_bias

        xh = np.tanh(xh) # activation function
        prev_hidden = xh
        hiddens[i,] = xh

        # output
        xo = xh @ o_weights + o_bias
        outputs[i] = xo

    y_true = [70, 62, 65]

    loss_grad = mse_grad(y_true, outputs)

    next_hidden = None

    o_weights_grad, o_bias_grad, h_weight_grad, h_bias_grad, i_weight_grad = [0] * 5

    for i in range(2, -1, -1):
        l_grad = loss_grad[i].reshape(1, 1) # l_grad shape is (1, 1)

        o_weights_grad += hiddens[i][:, np.newaxis] @ l_grad
        o_bias_grad += np.mean(l_grad)

        o_grad = l_grad @ o_weights.T

        if next_hidden is None:
            h_grad = o_grad
        else:
            h_grad = o_grad + next_hidden @ h_weights.T

        tanh_deriv = 1 - hiddens[i, :][np.newaxis, :]
        h_grad = h_grad * tanh_deriv

        next_hidden = h_grad

        if i > 0:
            h_weight_grad += hiddens[i-1][:,np.newaxis] @ h_grad
            h_bias_grad += np.mean(h_grad)

        i_weight_grad += sequence[i].reshape(1, 1).T * h_grad

    lr = 0.001
    i_weights -= i_weight_grad * lr
    h_weights -= h_weight_grad * lr
    h_bias -= h_bias_grad * lr
    o_weights -= o_weights_grad * lr
    o_bias -= o_bias_grad * lr
