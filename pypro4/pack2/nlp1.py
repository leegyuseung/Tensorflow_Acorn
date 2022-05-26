# 자연어 처리
# Word embedding : 워드를 수치화해서 벡터화
# 카테고리컬 인코딩 : 레이블 인코딩(문자를 숫자화), 원핫 인코딩(0과 1의 조합) 

print('레이블 인코딩 -------')
datas = ['python', 'lan', 'program', 'computer', 'say']
datas.sort() # 정렬
print(datas)

values = []
for x in range(len(datas)):
    values.append(x)

print(values, ' ', type(values))

print('원핫 인코딩 ------')
import numpy as np
onehot = np.eye(len(datas))
print(onehot, ' ', type(onehot))

print('레이블 인코딩 : 클래스 사용-----')
from sklearn.preprocessing import LabelEncoder
datas = ['python', 'lan', 'program', 'computer', 'say']
encoder = LabelEncoder().fit(datas)
values = encoder.transform(datas)
print(values, ' ', type(values))
print(encoder.classes_)

print('원핫 인코딩 : 클래스 사용-----')
from sklearn.preprocessing import OneHotEncoder
labels = values.reshape(-1, 1)
print(labels, ' ',labels.shape)
onehot = OneHotEncoder().fit(labels)
onehotValues = onehot.transform(labels)
print(onehotValues.toarray())

# 밀집 표현 : 단어마다 고유한 일련번호를 매겨서 사용하는 것이 아니라, 유사한 단어들을 비슷한 방향과 힘의 벡터를 갖도록 변환해 사용
# word2vec : 단어의 의미를 다차원 공간에 실수로 벡터화하는 분산표현 기법, 단어 간 유사성을 표현 간능

from gensim.models import word2vec
sentence = [['python', 'lan', 'program', 'computer', 'say']]
model = word2vec.Word2Vec(sentence, vector_size=30, min_count=1) 
print(model)
word_vectors = model.wv
print('word_vectors:', word_vectors)
print('word_vectors_index:,',word_vectors.key_to_index)
print('word_vectors_index:,',word_vectors.key_to_index.keys())
print('word_vectors_index:,',word_vectors.key_to_index.values())
vocabs = word_vectors.key_to_index.keys()
word_vectors_list = [word_vectors[v] for v in vocabs ]
print(word_vectors_list[0], len(word_vectors_list[0])) #say
print(word_vectors_list[1], len(word_vectors_list[1])) #computer

print()
# 단어 간 유사도(코사인 유사도) 확인 : 두 벡터가 닮은 정도를 정량적(-1 ~ 0 ~ 1)으로 나타낼 수 있다.
print(word_vectors.similarity(w1='python', w2='lan'))
print(word_vectors.similarity(w1='python', w2='say'))

# 시각화
import matplotlib.pyplot as plt
def plot_func(vocabs, x, y):
    plt.figure(figsize=(8,6))
    plt.scatter(x, y)
    for i, v in enumerate(vocabs):
        plt.annotate(v, xy=(x[i],y[i]))
    
from sklearn.decomposition import PCA # 주성분 분석
pca = PCA(n_components=2)
xys = pca.fit_transform(word_vectors_list)
xs = xys[:,0]
ys = xys[:,1]
plot_func(vocabs, xs, ys)
plt.show()