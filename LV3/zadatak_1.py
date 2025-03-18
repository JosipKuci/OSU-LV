import pandas as pd

data = pd.read_csv("data_C02_emission.csv")

data.columns = data.columns.str.strip()

def a():
    print(f"Broj mjerenja u dataframeu: {len(data)} \n")
    print("Tipovi podataka svih veličina:\n", data.dtypes)
    print(f"Broj izostalih vrijednosti: {data.isnull().sum().sum()} \n")
    print(f"Broj dupliciranih redaka: {data.duplicated().sum()}")

data = data.drop_duplicates().copy()
categorical_columns = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
for col in categorical_columns:
    data[col] = data[col].astype('category')

def b():
    print("3 vozila s najvećom gradskom potrošnjom goriva:\n")
    print(data.nlargest(3, 'Fuel Consumption City (L/100km)')[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

    print("3 vozila s najmanjom gradskom potrošnjom goriva:\n")
    print(data.nsmallest(3, 'Fuel Consumption City (L/100km)')[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

def c():
    engine_size = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
    print(f"\nBroj vozila s veličinom motora između 2.5 i 3.5 L: {len(engine_size)}")
    print(f"Prosječna emisija CO2 za ova vozila: {engine_size['CO2 Emissions (g/km)'].mean()} g/km")

def d():
    audi=data[data['Make']=='Audi']
    print(f"Broja audi vozila u datasetu: {len(audi)}\n")
    audi_4_cylinder=audi[audi['Cylinders']==4]
    print(f"Prosječna emisija CO2 plinova za 4 cilindrične Audi modele: {audi_4_cylinder['CO2 Emissions (g/km)'].mean()} g/km\n")

def e():
    #Trazenje najveceg cilindra
    print(data.nlargest(1,'Cylinders')[['Cylinders',]])

    cylinders4=data[data['Cylinders']==4]
    print(f"Broj vozila s 4 cilindra: {len(cylinders4)}\n  Njihova prosjecna potrosnja: {cylinders4['CO2 Emissions (g/km)'].mean()} g/km\n")

    cylinders6=data[data['Cylinders']==6]
    print(f"Broj vozila s 6 cilindra: {len(cylinders6)}\n  Njihova prosjecna potrosnja: {cylinders6['CO2 Emissions (g/km)'].mean()} g/km\n")

    cylinders8=data[data['Cylinders']==8]
    print(f"Broj vozila s 8 cilindra: {len(cylinders8)}\n  Njihova prosjecna potrosnja: {cylinders8['CO2 Emissions (g/km)'].mean()} g/km\n")

    cylinders10=data[data['Cylinders']==10]
    print(f"Broj vozila s 10 cilindra: {len(cylinders10)}\n  Njihova prosjecna potrosnja: {cylinders10['CO2 Emissions (g/km)'].mean()} g/km\n")

    cylinders12=data[data['Cylinders']==12]
    print(f"Broj vozila s 12 cilindra: {len(cylinders12)}\n  Njihova prosjecna potrosnja: {cylinders12['CO2 Emissions (g/km)'].mean()} g/km\n")


    #Izbačeno jer ih nema
    #cylinders14=data[data['Cylinders']==14]
    #print(f"Broj vozila s 14 cilindra: {len(cylinders14)}\n  Njihova prosjecna potrosnja: {cylinders14['CO2 Emissions (g/km)'].mean()} g/km\n")

    cylinders16=data[data['Cylinders']==16]
    print(f"Broj vozila s 16 cilindra: {len(cylinders16)}\n  Njihova prosjecna potrosnja: {cylinders16['CO2 Emissions (g/km)'].mean()} g/km\n")

def f():

    consumption_diesel = data[data['Fuel Type']=='D']
    consumption_gas = data[(data['Fuel Type']=='Z') | (data['Fuel Type']=='X')]
    print(f"Prosječna potrošnja dizelskih auta u gradu: {consumption_diesel['Fuel Consumption City (L/100km)'].mean()} L/100km")
    print(f"Prosječna potrošnja benzinskih auta u gradu: {consumption_gas['Fuel Consumption City (L/100km)'].mean()} L/100km")

    print(f"Medijalna potrošnja dizelskih auta u gradu: {consumption_diesel['Fuel Consumption City (L/100km)'].median()} L/100km")
    print(f"Medijalna potrošnja benzinskih auta u gradu: {consumption_gas['Fuel Consumption City (L/100km)'].median()} L/100km")

def g():
    cylinders4=data[data['Cylinders']==4]
    print(f"Vozilo s najvecom gradskom potrosnjom s 4 cilindra:\n {cylinders4.nlargest(1,'Fuel Consumption City (L/100km)')}")

def h():
    manual=data[(data['Transmission'].astype(str).str.contains("M"))]
    print(f"Broj auta s ručnim mjenjačem: {len(manual)}")
    
def i():
    #Pozitivna korelacija označava što veću korelaciju dok negativna najmanju, vidimo veliku korelaciju između veličine motora i broja cilindra
    correlation=data.corr(numeric_only=True)
    print(f"Korelacija numeričkih veličina: {correlation}")


a()
b()
c()
d()
e()
f()
g()
h()
i()