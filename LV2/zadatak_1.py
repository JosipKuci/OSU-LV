import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(1,3,num =3)
y1 = np.array([1,2,2],float)
plt.axis([0,4,0,4])

x2 = np.linspace(1,3,num =2)
y2 = np.array([1,1],float)


y3 = np.linspace(1,2,num =2)
x3 = np.array([3,3],float)

plt.plot(x1,y1,"b",linewidth=2, marker=".", markersize=10, color="yellow")
plt.plot(x2,y2,"b",linewidth=2, marker=".", markersize=10, color ="red")
plt.plot(x3,y3,"b",linewidth=2, marker=".", markersize=10, color="cyan")

plt.title("Malo se igramo")
plt.xlabel("x-os")
plt.ylabel("y-os")
plt.show()
