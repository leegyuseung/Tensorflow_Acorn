# 기존 소설로 자연어 생성 모델 작성 - 자소 단위 학습
# !pip install jamotools

import numpy as np
import tensorflow as tf
import jamotools

path_to_file = tf.keras.utils.get_file('toji.txt', 'https://raw.githubusercontent.com/pykwon/etc/master/rnn_short_toji.txt')
train_text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
s = train_text[:100]
print(s)

# 한글을 자모 단위로 분리
s_split = jamotools.split_syllables(s)
print(s_split)

# 다시 결합
s2 = jamotools.join_jamos(s_split)
print(s2)

# 자모 토큰 처리
train_text_x = jamotools.split_syllables(train_text)
vocab = sorted(set(train_text_x))
vocab.append('UNK')
print('사전에 등록된 자모 수 :', len(vocab))

# vocab의 자모를 숫자로 매핑
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
print(char2idx)
print(idx2char)
text_as_int = np.array([char2idx[c] for c in train_text_x])
print(text_as_int)
print()
print(train_text_x[:20]) #ㄱㅟㄴㅕㅇㅢ ㅁㅗㅅㅡㅂㅇㅡㄹ ㅎㅏㄴㅂ
print(text_as_int[:20]) #[35 80 38 70 56 83  2 50 72 54 82 51 56 82 43  2 63 64 38 51]

#기본 dataset 만들기
seq_length = 80 # 80개의 자소가 주어질 경우에 다음 자소를 예측하도록 ...
example_per_epoch = len(text_as_int) // seq_length
print(example_per_epoch) # 8636

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# seq_length + 1 : 80개 자소와 정답문자 1개를 합쳐 반환하기 위함
char_dataset = char_dataset.batch(seq_length + 1, drop_remainder=True) # 단일 원소를 n개 만큼 쌓기

for item in char_dataset.take(1):
    print(item.numpy()) # [35 80 38 70 56 83  2 50 72 54 
    print(idx2char[item.numpy()])  #['ㄱ' 'ㅟ' 'ㄴ' 'ㅕ' 'ㅇ' 'ㅢ' ' ' 'ㅁ
    print(len(item.numpy())) #81

# train dataset 작성 : 81개의 자소가 각각 입력과 정답으로 묶여 ([80자소], 1단어) 형태의 데이터 반환 함수
def split_input_target(chunk):
    return [chunk[:-1], chunk[-1]]  

train_dataset = char_dataset.map(split_input_target)
for x, y in train_dataset.take(1):
    print(idx2char[x.numpy()]) # ['ㄱ' 'ㅟ' 'ㄴ' 'ㅕ'
    print(x.numpy()) # 25개의 단어에 대한 숫자(단어표현) # [35 80 38 70 56 
    print(idx2char[y.numpy()]) # ㅇ
    print(y.numpy()) # 다음 단어 예측용 숫자  # 56

# dataset shuffle, batch 설정
BATCH_SIZE = 64
steps_per_epoch = example_per_epoch // BATCH_SIZE
BUFFER_SIZE =  5000
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# 자소 단위 모델
total_chars = len(vocab)

model = tf.keras.Sequential([
      tf.keras.layers.Embedding(total_chars, 100, input_length=seq_length),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.LSTM(units=400, activation='tanh'),
      tf.keras.layers.Dense(units=total_chars, activation='softmax')
])

model.compile(optimizer = 'adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# LamdaCallback : Lambda 함수 또는 사용자 함수를 작성하여 생성자에 넘기는 방식을 사용
# fit() 할 때 학습 도중 train data predict 되는 내용을 확인해 가며 작업 하고 싶을때 사용할 수도 있다
from keras.utils import pad_sequences
def testModel(epochs, logs):
    if epochs % 5 != 0 and epochs != 49:
        return
    
    test_sentence = train_text[:48]
    test_sentence = jamotools.split_syllables(test_sentence)
    next_chars = 300
    for _ in range(next_chars):
        test_text_x = test_sentence[-seq_length:]
        test_text_x = np.array([char2idx[c] if c in char2idx else char2idx['UNK'] for c in test_text_x])
        test_text_x = pad_sequences([test_text_x], maxlen=seq_length, padding='pre',value=char2idx['UNK'])
        output_idx = np.argmax(model.predict(test_text_x), axis=-1)
        test_sentence += idx2char[output_idx[0]]
    print()
    print(jamotools.join_jamos(test_sentence))
    print()

testModel_cb = tf.keras.callbacks.LambdaCallback(on_epoch_end=testModel)

#train_dataset.repeat() : 인풋 데이터 무한 반복 
#steps_per_epoch: epoch 당 사용할 strep 수를 지정. ex)총 45개의 sample이 있고 배치사이즈가 3이면, 15 step 지정
# 훈련데이터가 무한 크기인 경우 유용하다
history = model.fit(train_dataset.repeat(), epochs=50, steps_per_epoch=steps_per_epoch,
                    callbacks=[testModel_cb], verbose=2)

# 임의의 문장을 사용해 문자 생성
test_sentence='까까머리의 뒤통수는 골이 패인 것처럼 울퉁불퉁했다.'
next_chars = 500

for _ in range(next_chars):
    test_text_X = test_sentence[-seq_length:]
    test_text_X = np.array([char2idx[c] if c in char2idx else char2idx['UNK'] for c in test_text_X])
    test_text_X = pad_sequences([test_text_X], maxlen=seq_length, padding='pre', value=char2idx['UNK'])
    output_idx = np.argmax(model.predict(test_text_X), axis=-1)
    test_sentence += idx2char[output_idx[0]]
print()
print(test_sentence)
print()

# 시각화
import matplotlib.pyplot as plt
# %matplotlib inline

plt.plot(history.history['loss'], label='loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()