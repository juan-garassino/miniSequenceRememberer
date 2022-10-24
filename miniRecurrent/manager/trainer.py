from miniRecurrent.model.loss import cross_entropy
from miniRecurrent.model.optimizer import sgd
from miniRecurrent.model.model import net
from miniRecurrent.model.parameters import varaibles
from miniRecurrent.utils.utils import data_iter, tokenize, mapping, generate_training_data
from miniRecurrent.source.data import text, test

import os
from tensorflow import GradientTape

characters, vocabulary = tokenize(text)

(character_to_index, index_to_character), (word_to_index, index_to_word) = mapping(characters, vocabulary)

vocab_size = len(word_to_index)

print(vocab_size)

params = varaibles(vocab_size)

X, y = generate_training_data(vocabulary, word_to_index, 2, 42)

for epoch in range(int(os.environ.get('EPOCHS'))):
    for X, y in data_iter(int(os.environ.get('BATCH_SIZE')), X, y):
        # Compute gradients and update parameters
        with GradientTape() as tape:
            y_hat = net(X, params[0], params[1], vocab_size)
            l = cross_entropy(y_hat, y)
        # Compute gradient on l with respect to [`w`, `b`]
        grads = tape.gradient(l, params)
        # Update parameters using their gradient
        sgd(params, grads, float(os.environ.get('LEARNING_RATE')),
            int(os.environ.get('BATCH_SIZE')))

if __name__ == "__main__":
    pass
