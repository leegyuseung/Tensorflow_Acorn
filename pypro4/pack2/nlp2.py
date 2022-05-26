# 웹 뉴스 자료 읽어 형태소 분석 후 word2vec을 이용해 단어 간 유사도 확인하기
import pandas as pd
from konlpy.tag import Okt

# 형태소 분석
okt = Okt()

with open('news.txt', mode='r', encoding='utf-8') as f:
    lines = f.read().split('\n')
    
# print(lines)
# print(len(lines))

wordDic = {} # 형태소 분석 후 명사만 추출해 단어별 빈도수 => dict타입으로 확인

for line in lines:
    datas = okt.pos(line) #품사 태깅
    # print(datas)
    for word in datas:
        if word[1] == 'Noun':
            # print(word[0]) # Noun 뽑기
            if not(word[0] in wordDic):
                wordDic[word[0]] = 0
                        
            wordDic[word[0]] +=1
# print(wordDic)

keys = sorted(wordDic.items(), key = lambda x:x[1], reverse = True)
print(keys)

wordList = []
countList = []

for word, count in keys[:20]:
    wordList.append(word)
    countList.append(count)
    
df = pd.DataFrame()
df['word'] = wordList
df['count'] = countList
print(df)

print('--두 글자 이상의 데이터를 읽어 파일로 저장 ----------------')
result = []
results = []
with open('news.txt', mode='r', encoding='utf-8') as f:
    lines = f.read().split('\n')
    
for line in lines:
    datas = okt.pos(line, stem = True) #품사 태깅 # stem = True 원형 어근 출력. 한가한 -> 한가하다.
    imsi = []
    # print(datas)
    for word in datas:
        if not word[1] in ['Number','Alpha','Foreign','Josa','Punctuation','Determiner','Modifier']:
            if len(word) >= 2:
                imsi.append(word[0])
    # print(imsi)
    imsi2 = ("".join(imsi).strip())
    results.append(imsi2)
    
print(results)

fileName = 'news2.txt'
with open(fileName, 'w', encoding='utf-8') as fw:
    fw.write('\n'.join(results))

print('--- word2vec으로 밀집벡터를 만든 후 단어 유사도 확인 --- ')
from gensim.models import word2vec

lineObj = word2vec.LineSentence(fileName)

model = word2vec.Word2Vec(sentences=lineObj, vector_size=100, min_count=1, sg=0)
# sg = 0 : CBOW - 주변단어로 중심단어 예측, sg = 1 : Skip-gram - 중심단어로 주변단어 예측

print(model)

print(model.wv.most_similar(positive=['전기']))
print(model.wv.most_similar(positive=['전기'], topn=3))
print(model.wv.most_similar(positive=['전기', '메르세데스'], topn=3))
print(model.wv.most_similar(negative=['전기']))