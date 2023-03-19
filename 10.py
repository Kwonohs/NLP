from __future__ import division, print_function
import collections
import itertools
import nltk
import numpy as np
import matplotlib.pyplot as plt
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def get_data(Train_File):
    stories, questions, answers = [], [], []
    story_text = []
    fin = open(Train_File, "rb")
    for line in fin:
        line = line.decode("utf-8").strip()
        lno, text = line.split(" ", 1)
        if "\t" in text:
            question, answer, _ = text.split("\t")
            stories.append(story_text)
            questions.append(question)
            answers.append(answer)
            story_text = []

        else:
            story_text.append(text)
    fin.close()
    return stories, questions, answers

file_location_Train = "C:/Users/kwonohsem/Desktop/tasks_1-20_v1-2/en-10k/"
file_location_Test = "C:/Users/kwonohsem/Desktop/tasks_1-20_v1-2/en-10k/"
Train_File = os.path.join(file_location_Train, "qa1_single-supporting-fact_train.txt")
Test_File = os.path.join(file_location_Test, "qa1_single-supporting-fact_test.txt")
#데어터 가져오기
data_train = get_data(Train_File)
data_test = get_data(Test_File)
print("\n\nTrain observations:", len(data_train[0]), "Test observations:", len(data_test[0]),"\n\n")

dictnry = collections.Counter()
for stories, questions, answers in [data_train, data_test]:
    for story in stories:
        for sent in story:
            for word in nltk.word_tokenize(sent):
                dictnry[word.lower()] += 1
    for question in questions:
        for word in nltk.word_tokenize(question):
            dictnry[word.lower()] += 1
    for answer in answers:
        for word in nltk.word_tokenize(answer):
            dictnry[word.lower()] += 1
word2indx = {w:(i+1) for i,(w,_) in enumerate(dictnry.most_common())}
word2indx["PAD"] = 0
indx2word = {v:k for k,v in word2indx.items()}
vocab_size = len(word2indx)
print("vocabulary size:", len(word2indx))

story_maxlen = 0
question_maxlen = 0
for stories, questions, answers in [data_train, data_test]:
    for story in stories:
        story_len = 0
        for sent in story:
            swords = nltk.word_tokenize(sent)
            story_len += len(swords)
        if story_len > story_maxlen:
            story_maxlen = story_len
    for question in questions:
        question_len = len(nltk.word_tokenize(question))
        if question_len > question_maxlen:
            question_maxlen = question_len
print("Story maximum length:", story_maxlen, "Question maximum length:", question_maxlen)


from tensorflow.python.keras.layers import Input
from keras.layers.core import Activation, Dense, Dropout, Permute
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.merge import add, concatenate, dot
from tensorflow.python.keras.layers.recurrent import LSTM
from keras.models import Model
from keras.utils import pad_sequences, np_utils

def data_vectorization(data, word2indx, story_maxlen, question_maxlen):
    Xs, Xq, Y = [], [], []
    stories, questions, answers = data
    for story, question, answer in zip(stories, questions, answers):
        xs = [[word2indx[w.lower()] for w in nltk.word_tokenize(s)] for s in story]
        xs = list(itertools.chain.from_iterable(xs))
        xq = [word2indx[w.lower()] for w in nltk.word_tokenize(question)]
        Xs.append(xs)
        Xq.append(xq)
        Y.append(word2indx[answer.lower()])
    return pad_sequences(Xs, maxlen = story_maxlen), pad_sequences(Xq, maxlen = question_maxlen), np_utils.to_categorical(Y, num_classes = len(word2indx), dtype = object)

Xstrain, Xqtrain, Ytrain = data_vectorization(data_train, word2indx, story_maxlen, question_maxlen)
Xstest, Xqtest, Ytest = data_vectorization(data_test, word2indx, story_maxlen, question_maxlen)
print("Train story", Xstrain.shape, "Train question", Xqtrain.shape, "Train answer", Ytrain.shape)
print("Test story", Xstest.shape, "atEST QUESTION", Xqtest.shape, "Test answer", Ytest.shape)

EMBEDDING_SIZE = 128
LATENT_SIZE = 64
BATCH_SIZE = 64
NUM_EPOCHS = 40

story_input = Input(shape=(story_maxlen,))
question_input = Input(shape=(question_maxlen,))

story_encoder = Embedding(input_dim = vocab_size, output_dim = EMBEDDING_SIZE, input_length = story_maxlen) (story_input)
story_encoder = Dropout(0.2)(story_encoder)

question_encoder = Embedding(input_dim = vocab_size, output_dim = EMBEDDING_SIZE, input_length = question_maxlen) (question_input)
question_encoder = Dropout(0.3)(question_encoder)
match = dot([story_encoder, question_encoder], axes=[2, 2])
story_encoder_c = Embedding(input_dim = vocab_size, output_dim = question_maxlen, input_length = story_maxlen) (story_input)
story_encoder_c = Dropout(0.3)(story_encoder_c)

response = add([match, story_encoder_c])
response = Permute((2, 1))(response)

answer = concatenate([response, question_encoder], axis = -1)
answer = LSTM(LATENT_SIZE)(answer)
answer = Dropout(0.2)(answer)
answer = Dense(vocab_size)(answer)

output = Activation("softmax")(answer)
model = Model(inputs = [story_input, question_input], outputs = output)
model.compile(optimizer = "adma", loss = "categorical_crossentropy", metrics = ["accuracy"])
model.summary()