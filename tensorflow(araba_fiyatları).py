import pandas as pd 
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential #Burada modeli oluşturuyoruz 
from keras.layers import Dense #Burada model içine katmanları koyuyoruz
from sklearn.metrics import mean_absolute_error, mean_squared_error

dataFrame = pd.read_excel("merc.xlsx")

dataFrame.head() # Head ile verinin ilk 5 değeri gösterilir
dataFrame.describe()

dataFrame.isnull().sum() # boş null veri varmı ona bakar 

# ------------------ GRAFİKSEL ANALİZ ----------------------------------

sbn.displot(dataFrame["price"])
#plt.show()

sbn.countplot(dataFrame["year"])
#plt.show()

print(dataFrame.corr()) # verilerin birbirleriyle ilgili korelasyonu
print(dataFrame.corr()["price"].sort_values) # fiyatları küçükten büyüğe diz 

sbn.scatterplot(x="mileage", y="price", data=dataFrame)
#plt.show()

# ------------------ EN YÜKESEK FİYATLI ARABALAR VE VERİ TEMİZLİĞİ----------------------------------

dataFrame.sort_values("price", ascending=False).head(20) #ascending=false ifadesi en hüksek fiyatı en yukarıda getirecek
len(dataFrame)*0.01 #veri setinden atacağımız %1 lik kısımda kaç araç olduğunu buluruz 

yuzdeDoksanDokuzDf = dataFrame.sort_values("price", ascending=False).iloc[131:] # iloc[131:] ifadesi ile en yüksek fiyatlı 131 araba gitti bu da iloc(indexe göre sıralama da 131'den sonrasını al demiş olduk )
yuzdeDoksanDokuzDf.describe()

plt.figure(figsize=(7,5))
sbn.displot(yuzdeDoksanDokuzDf["price"])
#plt.show()

print(dataFrame.groupby("year").mean()["price"])
print(yuzdeDoksanDokuzDf.groupby("year").mean()["price"])

dataFrame = dataFrame[dataFrame.year != 1970] # 1970 yılındaki arabalarını sildik 
print(dataFrame.groupby("year").mean()["price"])

dataFrame = dataFrame.drop("transmission", axis=1)

# ------------------ EĞİTİM KISMI ----------------------------------

y = dataFrame["price"].values
x = dataFrame.drop("price",axis=1).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#x_train.shape

model = Sequential()
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

model.fit(x=x_train, y = y_train,validation_data=(x_test,y_test), batch_size=250, epochs=300) #validation_data ile bisiklet verisinde yaptığımız x_test ve y_test kıyaslama işlemini kendi yapacak 

# ------------------ DEĞERLENDİRME KISMI ----------------------------------

kayipVerisi = pd.DataFrame(model.history.history)
print(kayipVerisi.head())
kayipVerisi.plot()
plt.show() # buradan overfitting olup olmadığına bakabiliriz

tahminDizisi = model.predict(x_test)
print(mean_absolute_error(y_test,tahminDizisi))


#deneme ile sapma bakma ve tahmin etme (veri setinden çıkartıp değeri verdik ki değer ne kadar doğru ona baktık)
dataFrame.iloc[2]
yeniArabaSeries = dataFrame.drop("price", axis=1 ).iloc[2]

yeniArabaSeries =scaler.transform(yeniArabaSeries.values.reshape(-1,5))
print(model.predict(yeniArabaSeries))

