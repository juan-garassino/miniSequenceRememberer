from miniRecurrent.model.activation import softmax
from tensorflow import matmul, float32, reshape, cast


def embedding_net(X, params, vocab_size):
    X = reshape(cast(X, dtype=float32), (-1, vocab_size))
    a1 = matmul(X, params[0])
    a2 = matmul(a1, params[1])
    return softmax(a2), params
