from sklearn import datasets
from sklearn . model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn . preprocessing import MinMaxScaler
import math

#a)
numericke_vrijednosti = ['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']
df=pd.read_csv('data_C02_emission.csv')
X=df[numericke_vrijednosti]
y=df['CO2 Emissions (g/km)']
X_train , X_test , y_train , y_test = train_test_split (X, y, test_size = 0.2, random_state =1)

#b)
cylinders_test = X_test['Fuel Consumption Comb (mpg)']
co2_test=y_test
cylinders_train = X_train['Fuel Consumption Comb (mpg)']
co2_train=y_train
plt.scatter(cylinders_train,co2_train,s=3,alpha=0.7, c='blue')
plt.scatter(cylinders_test,co2_test,s=3,alpha=0.7, c='red')
plt.xlabel("Fuel Consumption Comb (mpg)")
plt.ylabel("CO2 Emissions (g/km)")
plt.show()

#c)
sc = MinMaxScaler ()
X_train_n = sc.fit_transform( X_train )
X_test_n = sc.transform( X_test )
plt.hist(X_train['Engine Size (L)'],color='blue',alpha=0.3,label="Original")
plt.hist(X_train_n[:,0], color='red',label="Scaled")
plt.title("Original vs scaled")
plt.legend()
plt.show()

#d)
import sklearn . linear_model as lm
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
parameters=linearModel.coef_
#Vidimo da ima 6 parametara koji odgovaraju 6 ulaznih vrijednosti za svaki izlaz
print(f"Parametri linearnog modela: {parameters}")

#e)
y_test_p = linearModel.predict(X_test_n)
plt.hist(y_test,color='blue',alpha=0.5,label="Real value")
plt.hist(y_test_p,color='red',alpha=0.5,label="Predicted value")
plt.legend()
plt.xlabel("CO2 Emissions (g/km)")
plt.title("Real vs Predicted values")
plt.show()

#f)
from sklearn.metrics import mean_absolute_error,mean_squared_error
mae = mean_absolute_error(y_true=y_test,y_pred=y_test_p)
mse = mean_squared_error(y_true=y_test,y_pred=y_test_p) #default=True
rmse = math.sqrt(mse)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

#g)
#Povecanjem testnog skupa povecava se vrijednost evaluacijskih metrika, obrnuto također
