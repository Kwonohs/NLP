'''

import re
def stem(word):
    splits = re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$', word)
    setm = splits[0][0]
    return stem
raw = "Keep your friends close, but your enemies closer."
tokens = re.findall(r'\w+|\S\w*', raw)
print(tokens)

for t in tokens:
    print("'" + stem(t) + "'")

'''


'''


import re

raw = "I am big! It's the pictures that got small.00"
print(re.split(r' +', raw))
print(re.split(r'\W+', raw))
print(re.findall(r'\w+|\S\w*', raw))

'''

'''

import re

street = '21 Teheran Road'
print(re.sub("Road", "RD", street))

text = "Diwali is a festival of light, Holi is a festival of color!"
print(re.findall(r"\b\w{5}\b", text))


'''







'''


import re

url = "http://www.telegraph.co.uk/formula-1.2018.10.28.mexican-grand-prix-2017-time-does-start-tv-channel-odds-lewis1/"
date_regex = '.(\d{4}).(\d{1,2}).(\d{1,2}).'
url = "https://www.naver.com/2014-04-28/"
date_regex = '/(\d{4})-(\d{1,2})-(\d{1,2})'
print("URL에서 찾은 날짜:", re.findall(date_regex,url))

def is_allowd_specific_char(string):
    charRe = re.compile(r'[^a-zA-Z0-9.]')
    string = charRe.search(string)
    return not bool(string)

print(is_allowd_specific_char("abcdAef123450."))
print(is_allowd_specific_char("*&&^@a."))
'''

'''

import re
patterns = ["Tuffy", "Pie", "Loki"]
text = "Tuffy eats pie, Loki eats peas!"

for pattern in patterns:
    print('"%s"에서 "%s"검색 중 ->' %(text, pattern),)
    if re.search(pattern, text):
        print("찾았습니다.")
    else:
        print("못찾았네요")

text = "Diwali is a festival of lights, Holi is a festival of colors!"
pattern = 'festival'

for match in re.finditer(pattern, text):
    s = match.start()
    e = match.end()
    print('%d:%d에서 "%s"을(를) 찾았습니다.' %(s, e, text[s:e]))
'''




'''


import re

def text_match(text, patterns):
    if re.search(patterns, text):
        return('일치하는 항목을 찾았습니다.')
    else:
        return("일치하지 않음!")

print("테스트 패턴은 다음으로 시작하고 끝남")
print(text_match("abbc", "^a.*c$"))

print("단어로 시작함")
print(text_match("Tuffy eats pie, Loki eats peas!", "^\w+"))

print("단어와 선택적 문장부호로 끝남")
print(text_match("Tuffy eats pie, Loki eats peas!", "^\w+.*\S*?$"))

print("단어의 시작이나 끝이 아닌 문자가 포함된 단어 찾기")
print(text_match("Tuffy eats pie, Loki eats peas!", "\Bu\B"))
print(text_match("asd a asd", r"\ba\b"))
'''



'''

import re
def text_match(text, patterns):
    if re.search(patterns, text):
        return('일치하는 항목을 찾았습니다.')
    else:
        return("일치하지 않음!")


print(text_match("ac", "ab?"))
print(text_match("abc", "ab?"))
print(text_match("abbc", "ab?"), "\n")

print(text_match("ac", "ab*"))
print(text_match("abc", "ab*"))
print(text_match("abbc", "ab*"), "\n")

print(text_match("ac", "ab+"))
print(text_match("abc", "ab+"))
print(text_match("abbc", "ab+"))

print(text_match("abc", "ab{2}"))
print(text_match("abbbc", "ab{3,5}"))
'''

'''

from nltk.metrics.distance import edit_distance
def my_edit_distance(str1, str2):
    m = len(str1) + 1
    n = len(str2) + 1
    table = {}
    for i in range(m):
        table[i,0] = i
        print(i)
    for j in range(n): table[0,j] = j
    for i in range(1, 5):
        for j in range(1, 4):
            cost = 0 if str1[i-1] == str2[j-1] else 1
            table[i,j] = min(table[i, j-1] + 1, table[i-1, j] +1, table[i-1, j-1] + cost)
    return table[i,j]

print("Our Algorithm :", my_edit_distance("hand", "and"))
print("NLTK Algorithm :", edit_distance("hand", "and"))


'''

'''

import nltk
from nltk.corpus import gutenberg
#불용어 처리 : 중요하지만 자주쓰여서 굳이 분류가 필요하지 않은 토큰

print(gutenberg.fileids())
gb_words = gutenberg.words('bible-kjv.txt')
words_filetered = [e for e in gb_words if len(e) >= 3]

stopwords = nltk.corpus.stopwords.words('english')
words = [w for w in words_filetered if w.lower() not in stopwords]

fdistPlain = nltk.FreqDist(words)
fdist = nltk.FreqDist(gb_words)

print("Following are the most common 10 words in the bag")
print(fdist.most_common(10))
print("Following are the most common 10 words in the bag minus the stopwords")
print(fdistPlain.most_common(10))

'''





'''

from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer
#원형 복원 : 사전을 토대로 원형으로 복원하는것
raw = "My name is maximus Decimus Meridius, commander of the Armies of the North General of the Felix Legions and loyal servant to the true emperor, Marcus Aurelius. Father to a murdered son, husband to a murdered wife. And I will have my vengeance, in this life or the next."
tokens = word_tokenize(raw)

porter = PorterStemmer()
stems = [porter.stem(t) for t in tokens]
print(stems)

lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(t) for t in tokens]
print(lemmas)
'''

'''

from nltk import PorterStemmer, LancasterStemmer, word_tokenize
#스테머 : 어간을 만드는것
raw = "My name is maximus Decimus Meridius, commander of the Armies of the North General of the Felix Legions and loyal servant to the true emperor, Marcus Aurelius. Father to a murdered son, husband to a murdered wife. And I will have my vengeance, in this life or the next."
tokens = word_tokenize(raw)
#print(tokens)

porter = PorterStemmer()
pStems = [porter.stem(t) for t in tokens]
print(pStems)

lancaster = LancasterStemmer()
lStems = [lancaster.stem(t) for t in tokens]
print(lStems)

'''
'''

from nltk.tokenize import LineTokenizer, SpaceTokenizer, TweetTokenizer
from nltk import word_tokenize

lTokenizer = LineTokenizer()
print("Line tokenizer 출력 : ", lTokenizer.tokenize("My name is maximus Decimus Meridius, "
                                                  "commander of the Armies of the North General "
                                                  "of the Felix Legions and loyal servant to the "
                                                  "true emperor, Marcus Aurelius. \nFather to a "
                                                  "murdered son, husband to a murdered wife. \nAnd "
                                                  "I will have my vengeance, in this life or the next."))

rawText = "By 11 o'clock on Sunday, the doctor shall open the dispensary."
sTokenizer = SpaceTokenizer()
print("Space Tokenizer 출력 : ", sTokenizer.tokenize(rawText))

print("Word Tokenizer 출력: ", word_tokenize(rawText))

tTokenzier = TweetTokenizer()
print("Tweet Tokenizer 출력 : ", tTokenzier.tokenize("This is a cooool #dummysmiley: :-) :-P <3"))
'''
