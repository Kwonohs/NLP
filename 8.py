#고객 감정에 대한 분석 탐색
import nltk
import nltk.sentiment.util
import nltk.sentiment.sentiment_analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def mySentimentAnalyzer():
    def score_feedback(text):
        positive_words = ["love", "genuine", "liked"]
        if "_NEG" in " ".join(nltk.sentiment.util.mark_negation(text.split())):
            score = -1
        else:
            analysis = nltk.sentiment.util.extract_unigram_feats(text.split(), positive_words)
            if True in analysis.values():
                score = 1
            else:
                score = 0
        return score

    feedback = """i love the items in this shop, very genuine and quality is well maintained.
    I have visited this shop and had samosa, my friends liked it very much.
    ok average food in this shop.
    Fridays are very busy in this shop, do not place orders during this day."""
    print(" -- custom scorer --")
    for text in feedback.split("\n"):
        print("score = {} for >> {}".format(score_feedback(text), text))

def advancedSentimentAnalyzer():
    sentences = [
        ':)',
        ":(",
        "she is so :(",
        "I love the way cricket is played by the champions",
        "She neither likes coffee nor tea"
    ]
    senti = SentimentIntensityAnalyzer()
    print(" -- built-in intensity analyser --")
    for sentence in sentences:
        print("[{}]".format(sentence), end = " --> ")
        kvp = senti.polarity_scores(sentence)
        for k in kvp:
            print("{} = {}, ".format(k, kvp[k]), end="")
            print()

if __name__ == "__main__":
    advancedSentimentAnalyzer()
    mySentimentAnalyzer()


'''

import nltk
import nltk.sentiment.sentiment_analyzer
import nltk.sentiment.util


def wordBaseSentiment():
    positive_words = ["love", "hope", "joy"]
    text = "Rainfall this year brings lot of hope and joy to Farmers.".split()
    analysis = nltk.sentiment.util.extract_unigram_feats(text, positive_words)
    print(" -- single word sentiment --")
    print(analysis)

def multiWordBasedSentiment():
    word_sets = [("heavy", "rains"), ("flood", "bengaluru")]
    text = "heavy rains cause flash flooding in bengaluru".split()
    analysis = nltk.sentiment.util.extract_bigram_feats(text, word_sets)
    print(" -- multi word sentiment --")
    print(analysis)


def markNegativity():
        text = "Rainfall last year did not bring joy to Farmers".split()
        negation = nltk.sentiment.util.mark_negation(text)
        print(" -- negativity --")
        print(negation)

if __name__ == "__main__":
    wordBaseSentiment()
    multiWordBasedSentiment()
    markNegativity()
'''
'''#주제식별
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim import corpora, models
import nltk
import feedparser

class IdentifyingTopicExample:
    def getDocuments(self):
        url = "https://sports.yahoo.com/mlb/rss.xml"
        feed = feedparser.parse(url) #feed에 위에 언급된 url에 모든 문서를 다운로드한 딕셔너리 리스트를 저장한다.
        self.documents = []
        for entry in feed["entries"][:5]:
            text = entry["summary"]
            if 'ex' in text:
                continue
            self.documents.append(text)
            print("-- {}".format(text))
        print("INFO : Fetching documents from {} completed".format(url))

    def cleanDocuments(self):
        tokenizer = RegexpTokenizer(r'[a-zA-z]+')
        en_stop = set(stopwords.words('english')) #불용어처리 하는 함수
        self.cleaned = []
        for doc in self.documents:
            lowercase_doc = doc.lower()
            words = tokenizer.tokenize(lowercase_doc)
            non_stopped_words = [i for i in words if not i in en_stop] #불용어를 제외한 단어를 변수에 저장
            self.cleaned.append(non_stopped_words)
        print("INFO: Cleaning {} documents completed".format(len(self.documents)))

    def doLDA(self):
        dictionary = corpora.Dictionary(self.cleaned)
        corpus = [dictionary.doc2bow(cleandoc) for cleandoc in self.cleaned]
        print(corpus)
        ldamodel = models.ldamodel.LdaModel(corpus, num_topics = 2, id2word = dictionary) #토픽 수를 2로 정한 말뭉치 모델을 생성하고
        #id2word 파라미터를 사용해 어휘 크기를 설정하고 매핑한다.
        print(ldamodel.print_topics(num_topics = 2, num_words = 4))
        #각 토픽마다 네 단어를 포함하는 두 개의 토픽을 화면에 출력한다.

    def run(self):
        self.getDocuments()
        self.cleanDocuments()
        self.doLDA()

if __name__ == '__main__':
    topicExample = IdentifyingTopicExample()
    topicExample.run()
'''


'''#잘 정의된 파이프라인은 다음 세개의 작업을 처리한다
# 각 구성 요소를 통과하는 데이터의 입력 형식
# 각 구성 요소에서 나오는 데이터의 출력 형식
# 데이터 유입 및 유출 속도를 조절해 구성 요소 간에 데이터 흐름이 제어되는지 확인

import nltk
import threading # 단일 프로그램에서 경량 태스크를 생성하는 데 사용 되는 스레딩 라이브러리
import queue #멀티스레딩 프로그램에서 사용할 수 있는 대기열 라이브러리
import feedparser # RSS 피드 구문 분석(파싱) 라이브러리
import uuid #RFC-4122 기반 uuid 버전 1,3,4,5 생성 라이브러리

threads = [] #리스트 생성
queues = [queue.Queue(), queue.Queue()] #두개의 대기열 생성 -> 첫번째 대기열은 토큰화된 문장 저장, 두번째 대기열은 분석된 모든 품사 단어를 저장


def extractWords():
    url = "https://timesofindia.indiatimes.com/rssfeeds/1081479906.cms"
    feed = feedparser.parse(url)
    for entry in feed["entries"][:5]:
        text = entry["title"]
        if 'ex' in text:
            continue
        words = nltk.word_tokenize(text)
        data = {'uuid' : uuid.uuid4(), "input" : words} #두개의 딕셔너리를 인수로 받는 data변수 생성
        queues[0].put(data, True) #data변수를 queue[0]에 저장 True는 대기열이 가득 차면 스레드 중지
        print(">> {} : {}".format(data['uuid'], text))

def extractPOS():
    while True:
        if queues[0].empty():
            break
        else:
            data = queues[0].get()
            words = data['input']
            postags = nltk.pos_tag(words)
            queues[0].task_done()
            queues[1].put({'uuid' : data['uuid'], 'input' : postags}, True)


def extractNE():
    while True:
        if queues[1].empty():
            break

        else:
            data = queues[1].get()
            postags = data["input"]
            queues[1].task_done()
            chunks = nltk.ne_chunk(postags, binary=False)
            print(" << {} : ".format(data["uuid"]), end = '')
            for path in chunks:
                try:
                    label = path.label()
                    print(path, end=', ')
                except:
                    pass
            print()

def runProgram():
    e = threading.Thread(target = extractWords())
    e.start()
    threads.append(e)

    p = threading.Thread(target=extractPOS())
    p.start()
    threads.append(p)

    n = threading.Thread(target=extractNE())
    n.start()
    threads.append(n)
    queues[0].join()
    queues[1].join()

    for t in threads:
        t.join()

if __name__ == '__main__': # 메인 스레드와 함께 실행될 때 호출되는 코드
    runProgram()
'''