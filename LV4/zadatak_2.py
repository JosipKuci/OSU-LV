from sklearn import datasets
from sklearn . model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn . preprocessing import MinMaxScaler
import math
import numpy as np

#a)
numericke_vrijednosti = ['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']
df=pd.read_csv('data_C02_emission.csv')
X=df[numericke_vrijednosti]

fuel_type=df[['Fuel Type']]
from sklearn . preprocessing import OneHotEncoder
ohe = OneHotEncoder (drop='first',sparse_output=False)
fuel_encoded=ohe.fit_transform(fuel_type)
X_encoded=np.hstack([X,fuel_encoded])
y=df['CO2 Emissions (g/km)']
X_train , X_test , y_train , y_test = train_test_split (X_encoded, y, test_size = 0.2, random_state =1)





import sklearn . linear_model as lm

from sklearn . preprocessing import OneHotEncoder


linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)


y_test_p = linearModel.predict(X_test)
plt.hist(y_test,color='blue',alpha=0.5,label="Real value")
plt.hist(y_test_p,color='red',alpha=0.5,label="Predicted value")
plt.legend()
plt.xlabel("CO2 Emissions (g/km)")
plt.title("Real vs Predicted values")
plt.show()


from sklearn.metrics import mean_absolute_error,mean_squared_error,max_error
mae = mean_absolute_error(y_true=y_test,y_pred=y_test_p)
mse = mean_squared_error(y_true=y_test,y_pred=y_test_p) #default=True
rmse = math.sqrt(mse)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

maksimalna_pogreska = max_error(y_test,y_test_p)
print(f"Maksimalna pogreska: {maksimalna_pogreska}")

indeks_maksimalne_pogreske=np.argmax(np.abs(y_test_p-y_test))
print(f"Radi se o autu:{df['Make'][indeks_maksimalne_pogreske]} {df['Model'][indeks_maksimalne_pogreske]}")

#Vidimo manje pogreške nego kod prvog zadatka
