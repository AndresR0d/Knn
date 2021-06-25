import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.algorithms import value_counts
from sklearn import neighbors
from sklearn.model_selection import train_test_split

#Obtenemos datos para crear el data frame
datos = pd.read_csv('vg1.csv')
datos = datos.replace(np.nan,"0")
df = pd.DataFrame(datos)

#Crear un diccionario de Python el cual guarde el numero del genero y el genero
pred = dict(zip(datos.GenreEt.unique(),datos.Genre.unique()))
print(pred)
print("------------------------")
#Contamos y enlistamos todos los juegos.
print(datos['Genre'].value_counts())

X=datos[['NA_Sales','JP_Sales','Global_Sales']]
y = datos['GenreEt']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Mostramos las estadisticas que se obtiene al establecer los datos de entrenamiento
print(X_train.describe())

#Creando el algoritmo de clasificacion KNNeighbors

from sklearn.neighbors import KNeighborsClassifier
neighbors=np.arange(1,9)
#asignamos la exactitud de los datos guardandolos en una funcion del tamano de los vecinos
train_accuracy=np.empty(len(neighbors))
test_accuracy=np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    train_accuracy[i]=knn.score(X_train,y_train)
    test_accuracy[i]=knn.score(X_test,y_test)


print("Clasificador KNN")
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
#Obtenemos la exactitud del algoritmo
print("Exactitud del algoritmo")
print(knn.score(X_test,y_test))
#Introducimos valores para nuestra prediccion
pred1=knn.predict([['6.38','4.46','6.04']])
print(pred[pred1[0]])

#------------------------------------------------
#Creamos algoritmo RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
print("Clasificador Random Forest")
print(("Exactitud del Algoritmo "))
rfc=RandomForestClassifier()
rfc.fit(X_train, y_train)
#Obtenemos la exactitud de RandomForest
print(rfc.score(X_test,y_test))
print("----------------------")
#Introducimos valores para nuestra prediccion
pred2=rfc.predict([['6.38','4.46','6.04']])
print(pred[pred2[0]])



#Creacion de grafica 

plt.title('Variacion vecinos en Knn')
plt.plot(neighbors,test_accuracy,label='Test accuracy')
plt.plot(neighbors,train_accuracy,label='Train accuracy')


plt.legend()
plt.xlabel('Vecinos/Neighbors')
plt.ylabel('Accuracy')
plt.show()


#Gracias :D
