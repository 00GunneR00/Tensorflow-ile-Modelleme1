import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sbn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping

dataFrame = pd.read_excel("maliciousornot.xlsx")
dataFrame.info()
dataFrame.describe()

dataFrame.corr()["Type"].sort_values() #tipe göre korelsayon
sbn.countplot(x="Type", data=dataFrame)
plt.show()

dataFrame.corr()["Type"].sort_values().plot(kind="bar") 
plt.show()

# ------------------------- SINIFLANDIRMA MODELİ --------------------------------------------

y = dataFrame["Type"].values
x = dataFrame.drop("Type", axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=15)

scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()

model.add(Dense(units=30, activation="relu")) #units 30 içeri 30 adet nöron koy demek 
model.add(Dense(units=15, activation="relu")) 
model.add(Dense(units=15, activation="relu")) 
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")

# ------------------------- TRAİN ETME --------------------------------------------

model.fit(x=x_train, y=y_train, epochs=700, validation_data=(x_test,y_test), verbose=1)

modelKaybi = pd.DataFrame(model.history.history)
modelKaybi.plot()
plt.show()

# ------------------------- EARLYSTOP --------------------------------------------

# earlystopping fonksiyonu val_loss kontrolü için kullanılır ve val_loss en kabulur edilir değere gelince duru (min)
model = Sequential()

model.add(Dense(units=30, activation="relu")) #units 30 içeri 30 adet nöron koy demek 
model.add(Dense(units=15, activation="relu")) 
model.add(Dense(units=15, activation="relu")) 
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")

earlyStopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=25)
model.fit(x=x_train, y=y_train, epochs=700, validation_data=(x_test,y_test), verbose=1, callbacks=[earlyStopping])

modelKaybi = pd.DataFrame(model.history.history)
modelKaybi.plot()
plt.show()

# ------------------------- DROPOUT LAYERS --------------------------------------------

# layer ile bir overfitting yaşanıyorsa en mutlak değere kadar layer atacak
#son katman hariç diğer tüm katmanlardan sonra eklenir

model = Sequential()

model.add(Dense(units=30, activation="relu")) #units 30 içeri 30 adet nöron koy demek 
model.add(Dropout(0.5))

model.add(Dense(units=15, activation="relu")) 
model.add(Dropout(0.5))

model.add(Dense(units=15, activation="relu")) 
model.add(Dropout(0.5))

model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")

model.fit(x=x_train, y=y_train, epochs=700, validation_data=(x_test,y_test), verbose=1, callbacks=[earlyStopping])


KayipDf = pd.DataFrame(model.history.history)
KayipDf.plot()
plt.show()

tahminlerimiz = model.predict_classes(x_test)

from sklearn.metrics import classification_report, confusion_matrix #sınıflandırmamızın ne kadar düzgün sonuç verdiğini değerlendirebiliriz 

print(classification_report(y_test,tahminlerimiz))
print(confusion_matrix(y_test,tahminlerimiz))
