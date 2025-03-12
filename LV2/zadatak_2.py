import numpy as np
import matplotlib.pyplot as plt
def b(data):
    weight=data[:,2]
    height=data[:,1]
    plt.scatter(weight,height)
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.show()

def c(data):
    weight=data[::50,2]
    height=data[::50,1]
    plt.scatter(weight,height)
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.title("Svaki 50-ti")
    plt.show()


data = np.genfromtxt('data.csv', delimiter=',', dtype=float, skip_header=1)
print(f"Number of participants: {data.shape[0]}") #A
#b(data) #B
#c(data)
height=data[:,1]
print(f"Min height: {height.min()}, Max height: {height.max()}, mean height:{height.mean()}")
men_height=[]
women_height=[]
gender=data[:,0]
index=0
for x in gender:
    if(x==0):
        women_height.append(height[index])
    elif(x==1):
        men_height.append(height[index])
    index+=1
women_height=np.asarray(women_height,dtype=float)
men_height=np.asarray(men_height,dtype=float)
print(f"WOMEN: Min height: {women_height.min()}, Max height: {women_height.max()}, mean height:{women_height.mean()}")
print(f"MEN: Min height: {men_height.min()}, Max height: {men_height.max()}, mean height:{men_height.mean()}")
