from tensorflow import Variable, random
import os

def varaibles(vocab_size):
    W1 = Variable(
        random.normal(shape=(vocab_size, int(os.environ.get('NUM_HIDDEN'))),
                      mean=0,
                      stddev=0.01))
    W2 = Variable(
        random.normal(shape=(int(os.environ.get('NUM_HIDDEN')), vocab_size),
                      mean=0,
                      stddev=0.01))

    return [W1, W2]
