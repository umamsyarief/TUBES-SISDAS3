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

#membuat list of K for KNN
k_list = list(range(1,50,2))

cv_scores = []

#Training KNN Classification Model
from sklearn.neighbors import KNeighborsClassifier

# melakukan 5-fold cross validation dengan k_list dalam knn
from sklearn.model_selection import cross_val_score
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    #Menentukan Prediksi dari X_test
    y_pred = knn.predict(X_test)
    # Menentukan probabilitas hasil prediksi
    knn.predict_proba(X_test)
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
    output1.to_excel(writer, sheet_name="K=1", index=False)
    output2.to_excel(writer, sheet_name="K=2", index=False)
    output3.to_excel(writer, sheet_name="K=3", index=False)
    output4.to_excel(writer, sheet_name="K=4", index=False)
    output5.to_excel(writer, sheet_name="K=5", index=False)