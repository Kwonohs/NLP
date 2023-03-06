import nltk
import pickle
def sampleData():
    return [
        "Bangalore is the capital of Karnataka.",
        "Steve Jobs was the CEO of Apple.",
        "iPhone was Invented by Apple",
        "Books can be purchased in Market."
    ]

def buildDictionary():
    dictionary = {}
    for sent in sampleData():
        partsOfSpeechTags = nltk.pos_tag(nltk.word_tokenize(sent))
        for tag in partsOfSpeechTags:
            value = tag[0]
            pos = tag[1]
            dictionary[value] = pos
    return dictionary

def saveMyTagger(tagger, fileName):
    fileHandle = open(fileName, "wb")
    pickle.dump(tagger, fileHandle)
    fileHandle.close()

def saveMyTraining(fileName):
    tagger = nltk.UnigramTagger(model = buildDictionary())
    saveMyTagger(tagger, fileName)

def loadMyTagger(fileName):
    return pickle.load(open(fileName, "rb"))

sentence = "Iphone is purchased by Steve Jobs in Bangalore Market"
fileName = "myTagger.pickle"
saveMyTraining(fileName)

myTagger = loadMyTagger(fileName)

print(myTagger.tag(nltk.word_tokenize(sentence)))

'''import nltk
simpleSentence = "Bangalore is the capital of Karnataka."
wordsInSentence = nltk.word_tokenize(simpleSentence)
print(wordsInSentence)
partsOfSpeechTags = nltk.pos_tag(wordsInSentence)
print(partsOfSpeechTags)
'''
'''import re

patterns = ["Tuffy", "Pie", "Loki"]
text = "Tuffy eats pie, Loki eats peas!"

for pattern in patterns:
    print("%s에서 %s 검색중 -> " %(text, pattern))
    if re.search(pattern, text):
        print("찾았습니다.!")
    else:
        print("못찾았네요")

text = "Diwali is a festival of lights, Holi is a festival of colors!"
pattern = "festival"

for match in re.finditer(pattern, text): #finditer함수는 text에 pattern과 일치
    #하는 문자를 찾아낸 위치를 반환한다.
    s = match.start() #s는 첫번째 festival에 위치를 반환받고 e는 마지막 위치를 반환
    e = match.end() #받는다.
    print("%d:%d에서 %s을(를) 찾았습니다." %(s, e, text[s:e]))'''