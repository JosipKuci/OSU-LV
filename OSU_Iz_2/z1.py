import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("pima-indians-diabetes.csv",delimiter=",", dtype=float)
print(f'Broj osoba na kojima su izvršena mjerenja{len(data)}')

data = [list(i) for i in data]
data = np.unique(data, axis=0)
bmi=data[:,5]
data = data[bmi!=0]

# Extract Age (column 7) and BMI (column 5)
age_bmi = data[:, [7, 5]]  

# Find unique (age, bmi) row combinations
_, unique_indices = np.unique(age_bmi, axis=0, return_index=True)

# Keep only rows with unique (age, bmi)
filtered_data = data[sorted(unique_indices)]
print(filtered_data)

print(f'Broj unosa nakon filtriranja: {len(filtered_data)}')

# c)
plt.scatter(filtered_data[:,7],filtered_data[:,5],s=1)
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('BMI by age')
plt.show()

# d)
print(f'Minimalna: {filtered_data[:,5].min()}')
print(f'Maksimalna: {filtered_data[:,5].max()}')
print(f'Srednja vrijednost: {filtered_data[:,5].mean()}')

# e)

data_diabetes=filtered_data[:,8]
has_diabetes=filtered_data[data_diabetes==1]
is_healthy=filtered_data[data_diabetes==0]
print('--------Imaju dijabetes---------')
print(f'Minimalna: {has_diabetes[:,5].min()}')
print(f'Maksimalna: {has_diabetes[:,5].max()}')
print(f'Srednja vrijednost: {has_diabetes[:,5].mean()}')

print('--------Nemaju dijabetes---------')
print(f'Minimalna: {is_healthy[:,5].min()}')
print(f'Maksimalna: {is_healthy[:,5].max()}')
print(f'Srednja vrijednost: {is_healthy[:,5].mean()}')

