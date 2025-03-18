import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")

data.columns = data.columns.str.strip()

def a():
    plt.figure(figsize=(8,5))
    plt.hist(data['CO2 Emissions (g/km)'],bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel("CO2 Emissions (g/km)")
    plt.ylabel("Broj vozila")
    plt.title("Emisija CO2 plinova")
    plt.grid(axis='y', linestyle='', alpha=0.7)
    plt.show()

def b():
    fuel_colors = {'X': 'green', 'Z': 'yellow', 'D': 'black', 'E': 'blue', 'N': 'orange'}
    plt.figure(figsize=(8,5))
    for fuel_type, color in fuel_colors.items():
        subset = data[data['Fuel Type'] == fuel_type]
        plt.scatter(subset['Fuel Consumption City (L/100km)'], subset['CO2 Emissions (g/km)'], c=color, label=fuel_type, alpha=0.5)
    plt.xlabel("Gradska potrošnja goriva (L/100km)")
    plt.ylabel("CO2 Emissions (g/km)")
    plt.title("Gradska potrošnja vs Emisija CO2 (po tipu goriva)")
    plt.legend(title="Tip goriva")
    plt.grid(True)
    plt.show()

def c():
    plt.figure(figsize=(8, 5))
    data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type', grid=False, vert=True, patch_artist=True)
    plt.xlabel("Tip goriva")
    plt.ylabel("Izvangradska potrošnja (L/100km)")
    plt.show()

def d():
    veichles_by_fuel_type = data['Fuel Type'].value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(veichles_by_fuel_type.index, veichles_by_fuel_type.values, color=['green', 'yellow', 'black', 'blue', 'orange'])
    plt.xlabel("Tip goriva")
    plt.ylabel("Broj vozila")
    plt.title("Broj vozila po tipu goriva")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def e():
    cylinder_avg_co2 = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
    plt.figure(figsize=(8, 5))
    plt.bar(cylinder_avg_co2.index.astype(str), cylinder_avg_co2.values, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel("Broj cilindara")
    plt.ylabel("Prosječna emisija CO2 (g/km)")
    plt.title("Prosječna emisija CO2 po broju cilindara")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


#a()
#b()
#c()
d()
e()