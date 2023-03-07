'''from nltk.grammar import CFG
from nltk.parse.chart import ChartParser, BU_LC_STRATEGY

grammar = CFG.fromstring("""
S -> T1 T4
T1 -> NNP VBZ
T2 -> DT NN
T3 -> IN NNP
T4 -> T3 | T2 T3
NNP -> "Tajmahal" | "Agra" | "Seoul" | | "Korea" | "and" | "HI" 
VBZ -> "is"
IN -> "in" | "of"
DT -> "the"
NN -> "capital"
""")

cp = ChartParser(grammar, BU_LC_STRATEGY, trace = True)

sentence = "Seoul is the capital of Korea"
sentence1 = "and HI"
tokens = sentence.split()
tokens1 = sentence1.split()

chart = cp.chart_parse(tokens)
chart1 = cp.chart_parse(tokens1)
parses = list(chart.parses(grammar.start()))
parses1 = list(chart1.parses(grammar.start()))
print("Total Edges :", len(chart.edges()))
print("Total Edges :", len(chart1.edges()))

for tree in parses: print(tree)
tree.draw()

'''

'''import nltk
def RDParserExample(grammar, textlist):
    parser = nltk.parse.RecursiveDescentParser(grammar)
    for text in textlist:
        sentence = nltk.word_tokenize(text)
        for tree in parser.parse(sentence):
            print(tree)
            tree.draw()


grammar = nltk.CFG.fromstring("""
S -> NP VP
NP -> NNP VBZ
VP -> IN NNP | DT NN IN NNP
NNP -> "Tajmahal" | "Agra" | "Bangalore" | "Karanataka"
VBZ -> "is"
IN -> "in" | "of"
DT -> "the"
NN -> "capital"
""")

text = [ "Tajmahal is in Agra",
         "Bangalore is the capital of Karnataka",]
RDParserExample(grammar, text)


'''
'''
import nltk
from nltk.corpus import conll2000 #conll2000 말뭉치 임포트
from nltk.corpus import treebank_chunk #말뭉치 임포트

def mySimpleChunker(): #NNP(고유명사)가 있는 모든 단어를 추출하는 태그패턴으로 청커가 개체명을 추출하는데 사용됨
    grammar = "NP : {<NNP>+} "
    return nltk.RegexpParser(grammar)

def test_nothing(data):
    cp = nltk.RegexpParser("")
    print(cp.accuracy(data))

def test_mysimplechunker(data, conll200=None):
    schunker = mySimpleChunker()
    print(schunker.accuracy(data))

datasets = [conll2000.chunked_sents('test.txt', chunk_types=["NP"]),treebank_chunk.chunked_sents()]

for dataset in datasets:
    test_nothing(dataset[:50])
    test_mysimplechunker(dataset[:50])
'''

'''import nltk
text = "Namsan Botanical Garden is a well known botanical gardenin Seoul, Korea."
sentences = nltk.sent_tokenize(text)
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    tags = nltk.pos_tag(words)
    chunks = nltk.ne_chunk(tags)
    print(chunks)
'''