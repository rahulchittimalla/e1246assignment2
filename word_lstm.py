
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

filename = "train.txt"
raw_text = open(filename).read()


def read(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

def vocab(filename):
    data = read(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def transform(filename, word_to_id):
    data = read(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

word_to_id = vocab(filename)
word_ids = transform(filename, word_to_id)


n_chars = len(word_ids)
n_vocab = len(word_to_id)


seq_length = 10
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = word_ids[i:i + seq_length]
	seq_out = word_ids[i + seq_length]
	dataX.append([word_to_id[word] for word in seq_in])
	dataY.append(word_to_id[seq_out])
n_patterns = len(dataX)


X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

X = X / float(n_vocab)

y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
