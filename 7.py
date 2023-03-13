import nltk

class LearningDictionary():
    def __init__(self, sentence):
        self.words = nltk.word_tokenize(sentence)
        self.tagged = nltk.pos_tag(self.words)
        self.buildDictionary()
        self.buildReverseDictionary()

    def buildDictionary(self):
        self.dictionary = {}
        for (word, pos) in self.tagged: #tag는 품사라고 보면 될듯
            self.dictionary[word] = pos
            print("하고가자" ,word,pos)


    def buildReverseDictionary(self):
        self.rdictionary = {}
        for key in self.dictionary.keys():
            value = self.dictionary[key]
            if value not in self.rdictionary:
                self.rdictionary[value] = [key]
            else:
                self.rdictionary[value].append(key)


    def isWordPresent(self, word):
        return "Yes" if word in self.dictionary else "No"

    def getPOSForWord(self, word):
        return self.dictionary[word] if word in self.dictionary else None

    def getWordsForPOS(self, pos):
        return self.rdictionary[pos] if pos in self.rdictionary else None

sentence = "All the flights got delayed due to bad weather"
learning = LearningDictionary(sentence)
words = ["chair", "flights", "delayed", "pencil", "weather"]
pos = ["NN", "VBS", "NNS"]
for word in words:
    status = learning.isWordPresent(word)
    print("Is '{}' present in dictionary ? : '{}'".format(word, status))
    if status is True:
        print("\tPOS for '{}' is '{}'".format(word, learning.getPOSForWord(word)))
    for pword in pos:
        print("POS '{}' has '{}' words".format(pword, learning.getWordsForPOS(pword)))


''''#청크의 정의는 개체라고 보면 되려나?
import nltk

def sampleNE():
    sent = nltk.corpus.treebank.tagged_sents()[0]
    print(nltk.ne_chunk(sent))

def sampleNEZ():
    sent = nltk.corpus.treebank.tagged_sents()[0]
    print(nltk.ne_chunk(sent, binary = True))

if __name__ == '__main__':
    sampleNE()
    sampleNEZ()
'''