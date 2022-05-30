# LSTM 을 사용해 영문 텍스트 생성 : 단어 단위
# 뉴욕타이즈 뉴스 기사 제목 추출해 데이터로 사용
# 데이터 불러오기
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/articlesapril.csv")
print(df.head(3))
print(df.count)
print(df.columns)

# unknown 제거
print(df['headline'].head(3))
print(df['headline'].isnull().values.any())
print(df.headline.values) 
headline = []
headline.extend(list(df.headline.values))
print(headline[:10])
print(len(headline)) # 1234
if 'Unknown' in headline:print("Unknown 데이터 발견")
headline = [n for n in headline if n != 'Unknown'] # 언노운 제거
print(len(headline)) # 1234
print(headline[:10])

# 구두점, 콤마 제거, 공백 제거
print('Hi하이llo 가a나다123'.encode('ascii', errors='ignore').decode())

from string import punctuation
print(", python`s.".strip(punctuation))
print(", python`s.".strip(punctuation+' '))

#
def repre_func(s):
    s = s.encode('utf8').decode('ascii','ignore') #아스키 코드 처리
    return ''.join(c for c in s if c not in punctuation).lower() 

# sss = ['abc123,.하하GOOd']
# imsi = repre_func(sss)
# print(imsi)

text = [repre_func(x) for x in headline]
print(text[:5])
# ['Former N.F.L. Cheerleaders’ Settlement Offer: $1 and a Meeting With Goodell', 'E.P.A. to Unveil a New Rule. Its Effect: Less Science in Policymaking.', 'The New Noma, Explained', 'How a Bag of Texas Dirt  Became a Times Tradition', 'Is School a Place for Self-Expression?', 'Commuter Reprogramming', 'Ford Changed Leaders, Looking for a Lift. It’s Still Looking.', 'Romney Failed to Win at Utah Convention, But Few Believe He’s Doomed', 'Chain Reaction', 'He Forced the Vatican to Investigate Sex Abuse. Now He’s Meeting With Pope Francis.']
# ['former nfl cheerleaders settlement offer 1 and a meeting with goodell', 'epa to unveil a new rule its effect less science in policymaking', 'the new noma explained', 'how a bag of texas dirt  became a times tradition', 'is school a place for selfexpression']

# 단어 집합
from keras.preprocessing.text import Tokenizer
tok = Tokenizer()
tok.fit_on_texts(text)
vocab_size = len(tok.word_index) + 1
print('단어집합 크기는', vocab_size)

sentences = list()
for i  in text:
    enc = tok.texts_to_sequences([i])[0]
    for j in range(1, len(enc)):
        se = enc[:j+1]
        sentences.append(se)

print(sentences)
print('dict items:', tok.word_index.items())

index_to_word = {}
for key, value in tok.word_index.items():
    index_to_word[value] = key
print(index_to_word)
print(index_to_word[10])

max_len = max(len(i) for i in sentences)
print('max_len:',max_len) # 24

# vector padding
from keras.utils import pad_sequences
psequences = pad_sequences(sentences, maxlen=max_len, padding='pre')
print(psequences[:3])

import numpy as np
psequences = np.array(psequences)
x = psequences[:, :-1]
y = psequences[:, -1]
print()
print(x[:3])
print(y[:3])

# label : one-hot encoding
from keras.utils.np_utils import to_categorical
y = to_categorical(y, num_classes=vocab_size)
print(y[:3])

# model
from keras.layers import Dense, LSTM, Embedding, Flatten
from keras.models import Sequential

model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_len - 1))
model.add(LSTM(128, activation='tanh'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(x, y, epochs=100, verbose=2)

#문자 생성
def sequence_gen_text(model, t, current_word, n):
    init_word = current_word
    sentence = ''
    for _ in range(n):
        encoded = t.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=max_len-1, padding='pre') # 패딩
        result = np.argmax(model.predict(encoded, verbose=0),axis=-1)
        
        # 예측단어 찾기
        for word, index in t.word_index.items():
            print(word, index)
            if index == result:
                break
            
        current_word = current_word +' '+word
        sentence = sentence+' '+ word
            
    sentence = init_word + sentence
    return sentence

print(sequence_gen_text(model, tok, 'i',20))
print(sequence_gen_text(model, tok, 'how',20))
