# 네이버 영화 리뷰 데이터로 감성분류

import numpy as np
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd 

train_data = pd.read_table('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/ratings_train.txt')
print(train_data[:2])
print(train_data.shape) #(150000, 3)
test_data = pd.read_table('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/ratings_test.txt')
print(test_data[:2])
print(test_data.shape) #(50000, 3)
print(set(test_data.label)) # {0, 1} 부정 , 긍정

# 중복 데이터 확인
print(train_data.document.nunique()) # 146182

train_data.drop_duplicates(subset=['document'], inplace=True)
print(train_data.shape) # (146183, 3)

# label 열로 그룹화
print(train_data.groupby('label').size()) # 0부정:73342 1긍정:72841

# null 확인 후 해당 행 삭제
print(train_data.isnull().sum()) # document    1
train_data = train_data.dropna(how='any')
print(train_data.shape) #(146182, 3)

# 한글과 공백만 데이터로 처리
train_data.document = train_data.document.str.replace("[^가-힣 ]", "")
print(train_data[:2])

train_data.document.replace('',np.nan, inplace=True)
print(train_data.isnull().sum())
print(train_data.loc[train_data.document.isnull()])

# document 열 중 NaN인 행 삭제
train_data = train_data.dropna(how='any')
print(train_data.shape) #  (145663, 3)

print()
test_data.drop_duplicates(subset=['document'], inplace=True)
test_data.document = test_data.document.str.replace("[^가-힣 ]", "")
test_data.document.replace('',np.nan, inplace=True)
test_data = test_data.dropna(how='any')
print(test_data.shape)  # (48929, 3)

# 불용어(stopwords) - 문장에 자주 등장하는 자주 등장하는 단어이지만 문맥에 영향을 주지 않는 단어들
stopwords = ['와','하다','한','은','를','는','의','좀','잘','과','으로','에는','에','하는','이지만','조차','아','그러나','그리고','그래서']

# 형태소 분석
okt = Okt()

# train data
x_train = []
for sentence in train_data['document']:
    imsi = []
    imsi = okt.morphs(sentence, stem=True) # 형태소단위로 텍스트를 나눠준다. step=True : 어간추출
    imsi = [word for word in imsi if not word in stopwords] # 불용어 제거
    x_train.append(imsi)

print(x_train[:3])

# test data
x_test = []
for sentence in test_data['document']:
    imsi = []
    imsi = okt.morphs(sentence, stem=True) # 형태소단위로 텍스트를 나눠준다. step=True : 어간추출
    imsi = [word for word in imsi if not word in stopwords] # 불용어 제거
    x_test.append(imsi)

print(x_test[:3])

# word embedding : 정수 인코딩
tok = Tokenizer()
tok.fit_on_texts(x_train)
print(tok.word_index)

# 등장 빈도수가 3회 미만인 단어의 비중 확인
threshold = 3
total_cnt = len(tok.word_index)
rare_cnt = 0
total_freq = 0
rare_freq = 0 

for key, value in tok.word_counts.items():
    total_freq = total_freq + value
    if value < threshold:
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합 크기 : ', total_cnt)
print('희귀 단어 수 : ', rare_cnt)
print('단어 집합 크기 :', (rare_cnt / total_cnt) * 100)
print('전체 등장 빈도에서 희귀단어 비율 :', (rare_freq / total_freq) * 100)

# 희귀단어 비율이 1.7389이므로 희귀단어 갯수는 제거 (2글자 이하 단어)
vocab_size = total_cnt - rare_cnt + 2 # +2 하는 이유는 pad와 oov 토큰을 사용할 예정.
print('단어사전크기:', vocab_size) # 19221

# 토큰화 되어 있지 않은 경우 (단어 사전에 등록되지 않는 단어)에는 oov(out of vocabulary)로 처리. oov는 보통 1로 할당
tok = Tokenizer(vocab_size, oov_token='OOV')
tok.fit_on_texts(x_train)
x_train = tok.texts_to_sequences(x_train)
x_test = tok.texts_to_sequences(x_test)
print(x_train[:3])

# label 별도 보관
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])
print(y_train[:3])

# 빈 샘플(empty sample) 제거
# 전체 데이터에서 빈도수가 낮은 단어가 삭제되었다는 것은 빈도수가 낮은 단어만으로 구성되었던 샘플들은 빈(empty) 샘플이 되었다는 것을 의미합니다.
# 빈 샘플들은 어떤 레이블이 붙어있던 의미가 없으므로 빈 샘플들을 제거해주는 작업을 하겠습니다. 
# 각 샘플들의 길이를 확인해서 길이가 0인 샘플들의 인덱스를 받아오겠습니다
drop_train = [index for index, sentence in enumerate(x_train) if len(sentence) < 1]
print(drop_train)

# 빈 샘플들을 제거
X_train = np.delete(x_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))

def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if(len(sentence) <= max_len):
            count = count + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

max_len = 30
below_threshold_len(max_len, X_train)

# 전체 훈련 데이터 중 약 94%의 리뷰가 30이하의 길이를 가지는 것을 확인했습니다. 모든 샘플의 길이를 30으로 맞추겠습니다.

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
print(x_train[:10])

# model
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units, activation='tanh'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

def sentiment_predict(new_sentence):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
    encoded = tok.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
    score = float(model.predict(pad_new)) # 예측
    if(score > 0.5):
        int("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    else:
        int("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
        
sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ')
sentiment_predict('이 영화 핵노잼 ㅠㅠ')
sentiment_predict('이딴게 영화냐 ㅉㅉ')
sentiment_predict('감독 뭐하는 놈이냐?')
sentiment_predict('와 개쩐다 정말 세계관 최강자들의 영화다')