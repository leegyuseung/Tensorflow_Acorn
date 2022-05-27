# RNN을 이용한 텍스트 생성 : 문맥을 반영해서 다음 단어를 예측하고 텍스트를 생성하기

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
import numpy as np
from keras.layers import Embedding, Dense, LSTM, Flatten
from keras.models import Sequential

text = """경마장에 있는 말이 뛰고 있다
그의 말이 법이다
가는 말이 고와야 온는 말이 곱다"""

# word indexing
tok = Tokenizer()
tok.fit_on_texts([text]) #list type
encoded = tok.texts_to_sequences([text])[0]#list type
# print(encoded)
# print(tok.word_index)

vocab_size = len(tok.word_index) + 1 # 원핫 처리, embedding에 사용

# 훈련 데이터 만들기
sequences = list()
for line in text.split('\n'): #문장 토큰화
    enco = tok.texts_to_sequences([line])[0]
    print(enco)
    # 바로 다음 단어를 label로 사용하기 위해
    for i in range(1, len(enco)):
        sequ = enco[:i+1]
        print(sequ) # [2, 3]...
        sequences.append(sequ)
        
print('학습에 참여할 샘플 수 :%d'%len(sequences)) #11
print(sequences) #[[2, 3], [2, 3, 1], [2, 3, 1, 4], ..

print(max(len(i) for i in sequences)) # 모든 벡터 중에서 가장 길이가 긴 값 출력 

# 전체 각가의 벡터의 길이를 통일
max_len = max(len(i) for i in sequences)

# 패딩채우기
psequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print(psequences)

# 각 벡터의 마지막 요소(단어)를 레이블로 사용하기 위해 분리
x = psequences[:,:-1] #feature
y = psequences[:, -1] #label
print(x)
print(y)

# 레이블을 원핫 처리
y = to_categorical(y, num_classes=vocab_size)
print(y[:2])

# model
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_len-1))
model.add(LSTM(32, activation ='tanh'))
model.add(Flatten()) # FClayer
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=200, verbose=2)
print('model.evaluate', model.evaluate(x, y))

# 문자열 생성 함수
def sequence_gen_text(model, t, current_word, n):
    init_word = current_word
    sentence = ''
    for _ in range(n):
        encoded = t.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=max_len-1, padding='pre') # 패딩
        result = np.argmax(model.predict(encoded, verbose=0),axis=-1)
        
        # 예측단어 찾기
        for word, index in t.word_index.items():
            #print(word, index)
            if index == result:
                break
            
        current_word = current_word +' '+word
        sentence = sentence+' '+ word
            
    sentence = init_word + sentence
    return sentence

print(sequence_gen_text(model, tok, '경마', 1))
print(sequence_gen_text(model, tok, '그의', 2))
print(sequence_gen_text(model, tok, '가는', 3))
print(sequence_gen_text(model, tok, '경마장에', 4))