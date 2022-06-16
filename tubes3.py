from msilib.schema import IniLocator
import matplotlib
import pandas as pd

data = pd.read_excel('dataset.xlsx')
x = data[data.columns[2:786]]
y = data[data.columns[1]]
#print(x)
#print(y)

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

knn = KNeighborsClassifier(n_neighbors=1) # knn dimana k=1
knn.fit(x,y)

from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred)) #score prediksi

#nilai k = 5 karena '5' merupakan 20% dari jumlah sampel yang ada
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=False)

print('{} {:^61} {}'.format('Iteration', 'Training set observation', 'Testing set observations'))
for iteration, data in enumerate(kf.split(x)):
    print('{!s:^9} {} {!s:^25}'.format(iteration, data[0], data[1]))

from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy')
print(scores)
print(scores.mean())

# mencari k terbaik untuk KNN
# range k yang ingin dicari
k_range = range(2,6)
# membuat list kosong untuk scores
k_scores = []

# melakukan looping sebanyak nilai k
for k in k_range:
    # menjalankan KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    # mendapatkan cross_val_score untuk KNeighborsClassifier
    scores = cross_val_score(knn, x, y, cv = 5, scoring= 'accuracy')
    # menambahkan skor rata rata pada list k_scores
    k_scores.append(scores.mean())

print(k_scores)

# panjang list
print('Length og list', len(k_scores))
print('Max of list', max(k_scores))

# Lakukan Visualisasi
import matplotlib.pyplot as plt
#matplotlib inline

plt.plot(k_range, k_scores)
plt.xlabel('Nilai K untuk KNN')
plt.ylabel('Hasil')
plt.title('Prediksi dengan KNN')

'''
print(x_train.shape) #bentuk data dari x_train
print(x_test.shape) #bentuk data dari x_test 
print(y_pred) #hasil prediksi
print(y_test) #jawaban sebenarnya
'''

'''
import numpy as np
import pandas as pd
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

#membuat class KNN
class KNN:
    def __init__(self, k=3):
        self.K=k
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_prediksi = [self._prediksi(x) for x in X]
        return np.array(y_prediksi)

    def _prediksi(self, x):
        #hitung jarak kesemua data training
        jarak_titik = [self.jarak(x,x_train) for x_train in self.X_train]
        #urutkan berdasarkan jarak terdekat, ambil sejumlah K
        k_terbaik = np.argsort(jarak_titik)[:self.K]
        #ambil label k_terbaik
        label_k_terbaik = [self.y_train[i] for i in k_terbaik]
        #voting berdasarkan terbanyak
        hasil_voting = Counter(label_k_terbaik).most_common(1)
        return hasil_voting[0][0]

    def jarak(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

#import data iris
iris = pd.read_excel('dataset.xlsx')
print(iris)
X,y = iris.data, iris.target

X_Train, X_Test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#print(X_Train)
#print(y_train)
model = KNN(k=3)
model.train(X_Train, y_train)
hasil = model.predict(X_Test)
print(hasil)

akurasi = np.sum(hasil == y_test)/len(X_Test)

print("akurasi: ", akurasi)
'''