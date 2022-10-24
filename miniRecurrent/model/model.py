from miniRecurrent.model.activation import softmax
from tensorflow import matmul, float32, reshape, cast


def net(X, W1, W2, vocab_size):
    X = reshape(cast(X, dtype=float32), (-1, vocab_size))
    a1 = matmul(X, W1)
    a2 = matmul(a1, W2)
    return softmax(a2)
