import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential #Burada modeli oluşturuyoruz 
from keras.layers import Dense #Burada model içine katmanları koyuyoruz

#sequnetial 

dataFrame = pd.read_excel("bisiklet.xlsx")

# y=wx+b
# y->label
y = dataFrame["Fiyat"].values


# x->feature(özellik)
x = dataFrame[["BisikletOzellik1","BisikletOzellik2"]].values


# Train ve Test olarak böldü 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=15)


#SCALİNG(BOYUT DEĞİŞİMİ)
#Burada nöronlara vereceğimiz veri setini kğçğk bir hale getirilmeye çalışılır 

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#artık bunları alıp kendi modelimize vermeye hazırız 

model = Sequential() #modelimizi oluşturduk 


# 3 adet hidden layer oluşturduk 
model.add(Dense(4,activation="relu")) #bu yapı tamamen hidden layerimiz | Dense içindeki ise nöron sayımız 
model.add(Dense(4,activation="relu")) #bu yapı tamamen hidden layerimiz | Dense içindeki ise nöron sayımız 
model.add(Dense(4,activation="relu")) #bu yapı tamamen hidden layerimiz | Dense içindeki ise nöron sayımız 

#output nöronumuz 
model.add(Dense(1))

model.compile(optimizer = "rmsprop", loss = "mse") #Compile bütün bu yaptığımız işlemleri birleştiriyor ve çalışmaya hazır hale getiriyor | 

# --------------------------------- DEVAM ----------------------------------------------- #

model.fit(x_train, y_train, epochs=250)

loss = model.history.history["loss"] #loss değerlerini liste içinde gösteriyor
sns.lineplot(x=range(len(loss)),y=loss)
plt.show()

# evaluate o fonksiyondaki loss değerlerini verir
trainLoss=model.evaluate (x_train,y_train)
testLoss=model.evaluate(x_test,y_test)

#train ve test değerleri birbirine yakınsa doğru yoldayız 
print(trainLoss)
print(testLoss)

#tahmin kısmı
testTahminleri = model.predict(x_test)
tahminDf = pd.DataFrame(y_test,columns=["Gerçek Y"])

testTahminleri = pd.Series(testTahminleri.reshape(330))

tahminDf = pd.concat([tahminDf,testTahminleri], axis=1)

tahminDf.columns = ["Gerçek Y", "Tahmin Y"]

print(tahminDf)

sns.scatterplot(x = "Gerçek Y", y = "Tahmin Y", data=tahminDf)
plt.show() 

from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(tahminDf["Gerçek Y"],tahminDf["Tahmin Y"])
dataFrame.describe() 


# model tahminleri 
yeniBisikletOzellikleri = [[1753,1751]]

yeniBisikletOzellikleri = scaler.transform(yeniBisikletOzellikleri)
model.predict(yeniBisikletOzellikleri)

# model kayıt etme 

from keras.models import load_model
model.save("bisiklet_modeli.h5")
sonradanCgrilanModel = load_model("bisikler_modeli.h5")
sonradanCgrilanModel.predict(yeniBisikletOzellikleri)
