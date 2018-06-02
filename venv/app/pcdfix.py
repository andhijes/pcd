import pandas as pd
mangga = pd.read_csv('mangga.csv', delimiter=';')

"""# Classification

## **Exploration Data**
"""

#print(mangga.head())

#print(mangga.describe().transpose())

#print(mangga.shape)

"""## Train Test Split"""

Xclass = mangga.drop('Kematangan',axis=1)
Xclass = Xclass.drop('luas',axis=1)
Xclass = Xclass.drop('Berat',axis=1)
yclass = mangga['Kematangan']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xclass, yclass)
print()

"""## Preprocessing"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

"""## Training Model"""

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(5,5,),max_iter=500)
mlp.fit(X_train,y_train)

"""## Save Model"""
import pickle
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(mlp, open(filename, 'wb'))


"""## Prediction"""
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))


"""## Buat ngetest klasifikasi"""
#load model yang udah di save
mlp = pickle.load(open('finalized_model.sav', 'rb'))
#misal barisnya [[r_avg, g_avg, b_avg, hijau, hitam]]
testbaris = [[45.42237509758,46.6865241998439,13.7977100274688,0.966212919594067,0.0337870804059329]]
#di praproses
testbaris = scaler.transform(testbaris)
#diprediksi
predictions = mlp.predict(testbaris)
print("hasil kelas: ")
#hasil prediksi
print(predictions)


"""## Buat ngetest regresi"""
#misal input
luas = 0.1
#model
berat = -54.6064 + 3184.4924*luas
#hasil berat
print('hasil berat: ' )
print(berat)