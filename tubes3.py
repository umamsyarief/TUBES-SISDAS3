import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

#membaca file dataset.xlsx
def baca_file():
    data = pd.read_excel('dataset.xlsx')
    X = data[data.columns[2:786]]
    y = data[data.columns[1]]

    X = np.array(X)
    y = np.array(y)

    #mengubah menjadi array 1 dimensi
    y = y.flatten()
    return X, y

#pilihan metric kalkulasi
def opsi_metric():
    print('''
Masukan metric perhitungan
1. Euclidean
2. Manhattan
3. Minkowski

Masukkan nomor metric pilihan anda :''')

    #memilih opsi untuk metric kalkulasi
    masukan = int(input())
    if masukan == 1:
        masukan = 'euclidean'
    elif masukan == 2:
        masukan = 'manhattan'
    elif masukan == 3:
        masukan = 'minkowski'
    return masukan

#perhitungan knn
def hitung_knn(masukan):
    #Melakukan pembagian data latih dan data uji 80%/20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=3)

    #membuat list K untuk KNN
    k_list = list(range(1,6,1))

    #membuat list kosong untuk cv_scores
    cv_scores = []

    for k in k_list:
        #Training KNN Classification Model
        knn = KNeighborsClassifier(n_neighbors=k, metric=masukan)
        knn.fit(X_train, y_train)
        #Menentukan Prediksi dari X_test
        y_pred = knn.predict(X_test)
        # Menentukan probabilitas hasil prediksi
        knn.predict_proba(X_test)
        # melakukan 5-fold cross validation dengan k_list dalam knn
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())

    #membuat dataframe untuk k=1
    output1 = pd.DataFrame({
        'idData' : np.arange(1, len(y_test) + 1),
        'label aktual' : y_test,
        'prediksi' : y_pred,
        'label luaran' : np.NaN
        })
    output1.at[0,'label luaran'] = cv_scores[0]

    #membuat dataframe untuk k=2
    output2 = pd.DataFrame({
        'idData' : np.arange(1, len(y_test) + 1),
        'label aktual' : y_test,
        'prediksi' : y_pred,
        'label luaran' : np.NaN
        })
    output2.at[0,'label luaran'] = cv_scores[1]

    #membuat dataframe untuk k=3
    output3 = pd.DataFrame({
        'idData' : np.arange(1, len(y_test) + 1),
        'label aktual' : y_test,
        'prediksi' : y_pred,
        'label luaran' : np.NaN
        })
    output3.at[0,'label luaran'] = cv_scores[2]

    #membuat dataframe untuk k=4
    output4 = pd.DataFrame({
        'idData' : np.arange(1, len(y_test) + 1),
        'label aktual' : y_test,
        'prediksi' : y_pred,
        'label luaran' : np.NaN
        })
    output4.at[0,'label luaran'] = cv_scores[3]

    #membuat dataframe untuk k=5
    output5 = pd.DataFrame({
        'idData' : np.arange(1, len(y_test) + 1),
        'label aktual' : y_test,
        'prediksi' : y_pred,
        'label luaran' : np.NaN
        })
    output5.at[0,'label luaran'] = cv_scores[4]

    #menggabungkan setiap dataframe pada sheet yang berbeda dalam file OutputValidasi.xlsx
    with pd.ExcelWriter("OutputValidasi.xlsx") as writer:
        output1.to_excel(writer, sheet_name="K=1", index=False)
        output2.to_excel(writer, sheet_name="K=2", index=False)
        output3.to_excel(writer, sheet_name="K=3", index=False)
        output4.to_excel(writer, sheet_name="K=4", index=False)
        output5.to_excel(writer, sheet_name="K=5", index=False)

#Program Utama
if __name__== "__main__":
    #baca file excel
    X, y = baca_file()
    #memilih metric yang digunakan
    masukan = opsi_metric()
    #lakukan perhitungan knn
    hitung_knn(masukan)
