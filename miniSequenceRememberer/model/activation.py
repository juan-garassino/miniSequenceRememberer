from tensorflow import reduce_sum, exp

def softmax(X):
    X_exp = exp(X)
    partition = reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition
