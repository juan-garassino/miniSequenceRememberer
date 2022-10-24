from miniRecurrent.model.loss import cross_entropy
from miniRecurrent.model.optimizer import sgd
from miniRecurrent.model.custom_embedding import embedding_net
from miniRecurrent.model.parameters import varaibles
from miniRecurrent.utils.utils import data_iter, tokenize, mapping, generate_training_data, one_hot_encode, get_embedding
from miniRecurrent.source.data import text, test
from miniRecurrent.model.recurrent_net import recurrent_net

import os
from tensorflow import GradientTape
import numpy as np

characters, vocabulary = tokenize(test)

(character_to_index, index_to_character), (word_to_index, index_to_word) = mapping(characters, vocabulary)

vocab_size = len(word_to_index)

print(vocab_size)

params = varaibles(vocab_size)

print(params[0])

print(params[1])

X, y = generate_training_data(vocabulary, word_to_index, 2, 42)

for epoch in range(int(os.environ.get('EPOCHS'))):
    for X, y in data_iter(int(os.environ.get('BATCH_SIZE')), X, y):
        # Compute gradients and update parameters
        with GradientTape() as tape:
            y_hat = embedding_net(X, params, vocab_size)[0]
            loss = cross_entropy(y_hat, y)
        # Compute gradient on l with respect to [`w`, `b`]
        grads = tape.gradient(loss, params)
        # Update parameters using their gradient
        sgd(params, grads, float(os.environ.get('LEARNING_RATE')),
            int(os.environ.get('BATCH_SIZE')))

learning = one_hot_encode(word_to_index["learning"], len(word_to_index))

print(learning)

result = embedding_net([learning], params, vocab_size)[0][0]

print(result)

for word in (index_to_word[id] for id in np.argsort(result)[::-1]):
    print(word)

print(params[0])

print(params[1])

print(get_embedding(word_to_index, params, 'machine'))

if __name__ == "__main__":
    pass
