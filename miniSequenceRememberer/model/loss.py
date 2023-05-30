from tensorflow import math, boolean_mask

def cross_entropy(y_hat, y):
    return -math.log(boolean_mask(y_hat, y))
