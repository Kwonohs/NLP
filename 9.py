
'''
import pandas as pd
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from sklearn.metrics import accuracy_score , classification_report
print(pad_sequences)

from keras.preprocessing import sequence


max_features = 6000
max_length = 400

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train observations')
print(len(x_test), 'test observations')

wind = imdb.get_word_index()
revind = dict((v,k) for k,v in wind.items()) #iteritems 대신 items 메서드를 사용했는데, 파이썬 3.x 버전부터는 iteritems 함수 지원이 되지 않는다.
print(x_train[0])
print(y_train[0])

def decode(sent_list):
    new_words = []
    for i in sent_list:
        new_words.append(revind[i])
    comb_words = " ".join(new_words)
    return comb_words
print(decode(x_train[0]))

x_train = pad_sequences(x_train, maxlen = max_length)
x_test = pad_sequences(x_test, maxlen=max_length)
print('x_train shape:', x_train.shape)
print("x_test shape:", x_test.shape)

batch_size = 32
embedding_dims = 60
num_kernels = 260
kernel_size = 3
hidden_dims = 300
epochs = 3

model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length = max_length))
model.add(Dropout(0.2))
model.add(Conv1D(num_kernels, kernel_size, padding = 'valid', activation = 'relu', strides = 1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)
score = model.evaluate(x_test, y_test, batch_size = batch_size)

y_train_case = model.predict(x_train, batch_size = batch_size)
y_train_predclass =  y_train_case.argmax(axis = -1)
y_test_case = model.predict(x_test, batch_size = batch_size)
y_test_predclass = y_test_case.argmax(axis = -1)
y_train_predclass.shape = y_train.shape
y_test_predclass.shape = y_test.shape

print(("\n\nCNN 1D - Train accuracy:"), (round(accuracy_score(y_train, y_train_predclass),3)))
print("\nCNN 1D of Training data\n", classification_report(y_train,y_train_predclass))
print("\nCNN 1D - Train Confusion Matrix\n\n", pd.crosstab(y_train, y_train_predclass, rownames = ["Actuall"], colnames = ["Predicted"]))
print(("\nCNN 1D - Test accuracy:"), (round(accuracy_score(y_test, y_test_predclass),3)))
print("\nCNN 1D of Test data\n", classification_report(y_test, y_test_predclass))
print("\nCNN 1D - Test Confusion Matrix\n\n", pd.crosstab(y_test, y_test_predclass, rownames = ["Actuall"], colnames = ["Predicted"]))
'''
'''
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset = 'train')
newsgroups_test = fetch_20newsgroups(subset = 'test')
x_train = newsgroups_train.data
x_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target
print("20개 카테고리 전체 목록:")
print(newsgroups_train.target_names)
print("\n")
print("샘플 이메일:")
print(x_train[0])
print("샘플 타깃 카테고리:")
print(y_train[0])
print(newsgroups_train.target_names[y_train[0]])

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
from nltk import pos_tag
from nltk.stem import PorterStemmer


def preprocessing(text):
    text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())
    tokens = [word for sent in nltk.sent_tokenize(text2) for word in nltk.word_tokenize(sent)]
    tokens = [word.lower() for word in tokens]
    stopwds = stopwords.words("english")
    tokens = [token for token in tokens if token not in stopwds]
    tokens = [word for word in tokens if len(word) >= 3]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    tagged_corpus = pos_tag(tokens)  # pos_tag 함수는 명사에 대한 네가지 형태와 동사에 대한 여섯가지 형태로 품사를 반환한다.
    # NN(명사,일반,단수), NNP(명사,보통,복수형), VB(동사,원형), VBD(동사,과거), VBG(동사, 현재 분사), VBN(동사,과거 분사), VBP(동사, 현재 시제, 3인칭 단수가 아닌 형태)
    # VBZ(동사, 현재 시제, 3인치 단수형)
    Noun_tags = ["NN", "NNP", "NNPS", "NNS"]
    Verb_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    lemmatizer = WordNetLemmatizer()

    def prat_lemmatize(token, tag):
        if tag in Noun_tags:
            return lemmatizer.lemmatize(token, 'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(token, 'v')
        else:
            return lemmatizer.lemmatize(token, 'n')

    pre_proc_text = " ".join([prat_lemmatize(token, tag) for token, tag in tagged_corpus])
    return pre_proc_text


x_train_preprocessed = []
for i in x_train:
    x_train_preprocessed.append(preprocessing(i))
x_test_preprocessed = []
for i in x_test:
    x_test_preprocessed.append(preprocessing(i))
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 2, ngram_range = (1, 2), stop_words = "english", max_features = 10000, strip_accents = "unicode", norm = 12)
x_train_2 = vectorizer.fit_transform(x_train_preprocessed).todense()
x_test_2 = vectorizer.transform(x_test_preprocessed).todense()

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta, Adam, RMSprop
from keras.utils import np_utils

np.random.seed(1337)
nb_classes = 20
batch_size = 64
nb_epochs = 20

Y_train = np_utils.to_categorical(y_train, nb_classes)

model = Sequential()
model.add(Dense(1000, input_shape=(10000,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
print(model.summary())

model.fit(x_train_2, Y_train, batch_size = batch_size, epochs = nb_epochs, verbose = 1 )

y_train_predclass = model.predict_classes(x_train_2, batch_size = batch_size)
y_test_predclass = model.predict_classes(x_test_2, batch_size = batch_size)
from sklearn.metrics import accuracy_score, classification_report
print("\n\nDeep Neutral Network - Train accuracy:"), (round(accuracy_score(y_test, y_test_predclass),3))
print("\n Deep Neutral Netword - Train Classification Report")
print(classification_report(y_train, y_train_predclass))
print("\nDeep Neutral Netword - Test classification Report")
print(classification_report(y_test, y_test_predclass))


'''