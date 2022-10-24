from tensorflow.keras.utils import get_file

FILE_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

FILE_NAME = 'shakespeare.txt'

path_to_file = get_file('shakespeare.txt', FILE_URL)

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

print(f'Lenght of text is: {len(text)} characters')

test = '''Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks.'''
