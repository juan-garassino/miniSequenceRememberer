import numpy as np
import random
from tensorflow import constant, gather

def tokenize(text):

    text_lower = text.lower()

    vocabulary_char = sorted(set(text_lower))

    print(f'There are {len(vocabulary_char)} unique characters')

    vocabulary_word = sorted(set(text_lower.split(' ')))

    print(f'There are {len(vocabulary_word)} unique words')

    return vocabulary_char, vocabulary_word

def mapping(vocabulary_characters, vocabulary_word):

    character_to_index = {
        character: index
        for index, character in enumerate(vocabulary_characters)
    }

    word_to_index = {word:index for index, word in enumerate(vocabulary_word)}

    index_to_character = {
        index: character
        for index, character in enumerate(vocabulary_characters)
    }

    index_to_word = {index:word for index, word in enumerate(vocabulary_word)}

    return (character_to_index, index_to_character), (word_to_index,
                                                      index_to_word)


def concat(*iterables):
    for iterable in iterables:
        yield from iterable

def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res

def generate_training_data(tokens, word_to_id, window, seed):

    np.random.seed(seed)

    X = []
    y = []
    n_tokens = len(tokens)

    for i in range(n_tokens):
        idx = concat(range(max(0, i - window), i),
                     range(i, min(n_tokens, i + window + 1)))
        for j in idx:
            if i == j:
                continue
            X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
            y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))

    return np.asarray(X), np.asarray(y)


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = constant(indices[i:min(i + batch_size, num_examples)])
        yield gather(features, j), gather(labels, j)
