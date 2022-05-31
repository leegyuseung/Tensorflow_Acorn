# IMDB dataset : 미국 영화 리뷰 데이터
# 실습1) LSTM으로 이항분류
# 실습2) CNN으로 이항분류

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data()

print(X_train.shape,y_train.shape,X_test.shape, y_test.shape)
print('훈련용 리뷰 개수 : {}'.format(len(X_train)))
print('테스트용 리뷰 개수 : {}'.format(len(X_test)))
num_classes = len(set(y_train))
print('카테고리 : {}'.format(num_classes))

print('첫번째 훈련용 리뷰 :',X_train[0])
print('첫번째 훈련용 리뷰의 레이블 :',y_train[0])

len_result = [len(i) for i in X_train]

print('리뷰의 최대 길이 : {}'.format(np.max(len_result)))
print('리뷰의 평균 길이 : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

# 레이블의 분포
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("각 레이블에 대한 빈도수:")
print(np.asarray((unique_elements, counts_elements)))

word_to_index = imdb.get_word_index()
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value+3] = key

print(index_to_word)   
print('빈도수 상위 1등 단어 : {}'.format(index_to_word[4]))
print('빈도수 상위 3938등 단어 : {}'.format(index_to_word[3941]))

#첫번째 훈련용 리뷰의 X_train[0]의 각 단어가 정수로 바뀌기 전에 어떤 단어들이었는지 확인
for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index] = token

print(' '.join([index_to_word[index] for index in X_train[0]]))

#### IMDB 리뷰 감성 분류하기
import re
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

vocab_size = 10000
max_len = 500 # 최대 리뷰 길이 500
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# 실습1) LSTM으로 이항분류
model = Sequential()
model.add(Embedding(vocab_size, 200, input_length=max_len))
model.add(LSTM(128, activation='tanh'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='auto', verbose=0, patience=3, baseline=0.01)
mc = ModelCheckpoint('rnn_imdb_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history = model.fit(X_train, y_train, epochs=30, callbacks=[es, mc], batch_size=64, validation_split=0.2,
                    verbose = 2)
print('acc:',history.history['acc'])
print('loss:',history.history['loss'])
print('evaluate:',model.evaluate(X_test,y_test)[1])

# 실습2) CNN으로 이항분류
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dropout

cmodel = Sequential()
cmodel.add(Embedding(vocab_size, 200, input_length=max_len))
cmodel.add(Conv1D(filters=256,kernel_size=3, padding='valid',strides=1, activation='relu'))
cmodel.add(GlobalMaxPooling1D())
cmodel.add(Dropout(0.3))
cmodel.add(Dense(64, activation='relu'))
cmodel.add(Dropout(0.3))
cmodel.add(Dense(1, activation='sigmoid'))
print(cmodel.summary())

cmodel.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='auto', verbose=0, patience=3, baseline=0.01)
mc = ModelCheckpoint('rnn_imdb_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history = cmodel.fit(X_train, y_train, epochs=30, callbacks=[es, mc], batch_size=64, validation_split=0.2,
                    verbose = 2)
print('acc:',history.history['acc'])
print('loss:',history.history['loss'])
print('evaluate:',cmodel.evaluate(X_test,y_test)[1])

# 시각화
vloss = history.history['val_loss']
loss = history.history['loss']
epoch = np.arange(len(loss))
plt.plot(epoch, vloss, marker='s', c='r', label='val_loss')
plt.plot(epoch, loss, marker='o', c='b', label='loss')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.show()

# 예측
def sentiment_predict(new_sentence):
    # 알파벳과 숫자를 제외하고 모두 제거 및 알파벳 소문자화
    new_sentence = re.sub('[^0-9a-zA-Z ]', '', new_sentence).lower()
    encoded = []

    # 띄어쓰기 단위 토큰화 후 정수 인코딩
    for word in new_sentence.split():
        try :
        # 단어 집합의 크기를 10,000으로 제한.
            if word_to_index[word] <= 10000:
                encoded.append(word_to_index[word]+3)
            else:
            # 10,000 이상의 숫자는 <unk> 토큰으로 변환.
                encoded.append(2)
        # 단어 집합에 없는 단어는 <unk> 토큰으로 변환.
        except KeyError:
            encoded.append(2)

    pad_sequence = pad_sequences([encoded], maxlen=max_len)
    score = float(cmodel.predict(pad_sequence)) # 예측

    if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))

test_input = " I was lucky enough to be included in the group to see the advanced screening in Melbourne on the 15th of April, 2012. And, firstly, I need to say a big thank-you to Disney and Marvel Studios. \
Now, the film... how can I even begin to explain how I feel about this film? It is, as the title of this review says a 'comic book triumph'. I went into the film with very, very high expectations and I was not disappointed. \
Seeing Joss Whedon's direction and envisioning of the film come to life on the big screen is perfect. The script is amazingly detailed and laced with sharp wit a humor. The special effects are literally mind-blowing and the action scenes are both hard-hitting and beautifully choreographed."

sentiment_predict(test_input)
test_input = 'good'
sentiment_predict(test_input)
test_input = 'bad'
sentiment_predict(test_input)