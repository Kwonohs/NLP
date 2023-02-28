
'''
import nltk, matplotlib
from nltk.corpus import webtext
print(webtext.fileids())

fileid = 'singles.txt'
wbt_words = webtext.words(fileid)
fdist = nltk.FreqDist(wbt_words)
print("최대 발생 토근", fdist.max(), "수 : ", fdist[fdist.max()])
#fdist.max()는 빈도수가 가장 높은 단어를 출력 fdist[fidst.max()]는 가장높은 단어의 수를 출력
print('말뭉치 내 총 고유 토큰 수 : ', fdist.N())
print(fdist.most_common(10))
print(fdist.tabulate())
fdist.plot(cumulative=True)
'''

'''
import nltk
from nltk.corpus import brown
print(brown.categories())
genres = ['fiction', 'humor', 'romance']
whwords = ['what', 'which', 'how', 'why', 'when', 'where', 'who']

for i in range(0, len(genres)):
    genre = genres[i]
    print()
    print("'" + genre + "' wh 단어 분석")
    genre_text = brown.words(categories = genre)
    fdist = nltk.FreqDist(genre_text) #빈도수를 반환해주는 함수
    for wh in whwords:
        print(wh + ":", fdist[wh], end=' ')
'''
'''

from nltk.corpus import CategorizedPlaintextCorpusReader

reader = CategorizedPlaintextCorpusReader(r'/Users/kwonohsem/Desktop/mix20_rand700_tokens_cleaned/tokens', r'.*\.txt', cat_pattern = r'(\w+)/*')
print(reader.categories())
print(reader.fileids())

posFiles = reader.fileids(categories = 'pos')
negFiles = reader.fileids(categories = 'neg')

from random import randint
fileP = posFiles[randint(0, len(posFiles) - 1)]
fileN = negFiles[randint(0, len(negFiles) - 1)]
print(fileP)
print(fileN)

for w in reader.words(fileP):
    print(w + ' ', end = '')
    if w == '.':
        print()

for w in reader.words(fileN):
    print(w + ' ', end = '')
    if w == '.':
        print()
        '''


'''from nltk.corpus import reuters

files = reuters.fileids() # files변수에 reuters에서 가져온 데이터의 상대경로들을 저장한다.
#print(files)

words16097 = reuters.words(['test/16097'])
words16098 = reuters.words(['test/16098'])
words16099 = reuters.words(['test/16099'])
words20 = reuters.words(['test/16097'])#[20:] 뒤에 이런식으로 [:20] 해주는 표현법도 익히면 좋겠다.

#print(words16097)
#print(words16098)
#print(words16099)
print(words20[0:])


reutersGenres =  reuters.categories()
print(reutersGenres)

'''