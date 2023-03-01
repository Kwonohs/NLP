'''#2.단원에 pdf와 window ward 관련된 것들은 따로 공부해야한다.
#3.전처리
from nltk.tokenize import LineTokenizer, SpaceTokenizer, TweetTokenizer
from nltk import word_tokenize

lTokenizer = LineTokenizer()
a = "My name, is Maximus. \nI don't kwon you but I love you."
print("Line tokenizer 출력 : " ,lTokenizer.tokenize(a))
#줄 바꿈마다 리스트에 나뉘어서 저장되는 토크나이저
sTokenizer = SpaceTokenizer()
print("Space Tokenizer 출력: ", sTokenizer.tokenize(a))
#공백 문자마다 리스트에 나뉘어서 저장
print("ddd :", word_tokenize(a))
#word_toeknize는 따로 생성자가 없으니 직접 텍스트를 입력받아야한다.
#어절과, ',"과 같은 특수문자들이 나뉘어 리스트에 저장된다.


'''
'''#스테밍 : 어간을 추출하는것
#어간 : 접미사가 없는 단어의 기본형
#스테머 : 접미사를 제거하고 단어의 어간을 반환하는것

from nltk import PorterStemmer, LancasterStemmer, word_tokenize
a = "My name, is Maximus. \nI don't kwon you but I love you."
tokens = word_tokenize(a)
porter = PorterStemmer()
pStems = [porter.stem(t) for t in tokens] #for문 응용한꼴 기억하면 좋을듯
print(pStems)
#porterStemmer 보다 LancasterStemmer가 더 많은 접미사를 제거해주는 스테머이다.
'''

#lemma란 단어의 기본형을 나타낸다. 사전과 일치하는 기본 형태라고 생각하면 됨

from nltk import word_tokenize, PorterStemmer, WordNetLemmatizer
#wordNetLemmatizer을 활용하면 위에 스테머들보다 기본형을 얻는데 더욱 효율적이다.
#그러나 위에 스테머들은 단순하게 접미사를 제거하기에 속도측면에서는 효율적이다.
