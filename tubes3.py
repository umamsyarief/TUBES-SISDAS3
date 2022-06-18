from statistics import mode
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

data = pd.read_excel('dataset.xlsx')
X = data[data.columns[2:786]]
y = data[data.columns[1]]

X = np.array(X)
y = np.array(y)

#mengubah menjadi array 1 dimensi
y = y.flatten()

#Melakukan pembagian data latih dan data uji 80%/20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=3)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

#Training KNN Classification Model
from sklearn.neighbors import KNeighborsClassifier
#Nilai K pada KNN
K = 3
model = KNeighborsClassifier(n_neighbors=K)
model.fit(X_train, y_train)

#Menentukan Prediksi dari X_test
y_pred = model.predict(X_test)

# Menentukan probabilitas hasil prediksi
model.predict_proba(X_test)

#Menampilkan hasil akurasi
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test, y_pred)*100

print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

from sklearn.model_selection import cross_val_score

#membuat list of K for KNN
k_list = list(range(1,50,2))

cv_scores = []

# melakukan 5-fold cross validation dengan k = 1 dalam knn
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

output1 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
output1.at[0,'label luaran'] = cv_scores[0]

output2 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
output2.at[0,'label luaran'] = cv_scores[1]

output3 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
output3.at[0,'label luaran'] = cv_scores[2]

output4 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
output4.at[0,'label luaran'] = cv_scores[3]

output5 = pd.DataFrame({
    'idData' : np.arange(1, len(y_test) + 1),
    'label aktual' : y_test,
    'prediksi' : y_pred,
    'label luaran' : np.NaN
    })
output5.at[0,'label luaran'] = cv_scores[4]

with pd.ExcelWriter("OutputValidasi.xlsx") as writer:
    # use to_excel function and specify the sheet_name and index
    # to store the dataframe in specified sheet
    output1.to_excel(writer, sheet_name="K=1", index=False)
    output2.to_excel(writer, sheet_name="K=2", index=False)
    output3.to_excel(writer, sheet_name="K=3", index=False)
    output4.to_excel(writer, sheet_name="K=4", index=False)
    output5.to_excel(writer, sheet_name="K=5", index=False)