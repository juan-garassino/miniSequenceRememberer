def sgd(params, grads, lr, batch_size):
    """Minibatch stochastic gradient descent.
    Defined in :numref:`sec_linear_scratch`"""
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size)
