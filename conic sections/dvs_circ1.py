import numpy as np
import matplotlib.pyplot as plt

A = np.array((2,3))
B = np.array((4,5))
C = (A+B)/2
print(C)

m = A-B #direction vector of (A,B)
print(m)

n = np.array((1,1))
m = np.array((-1,4))
i = (n,m)
print(i)
c1 = 7
c2 = 3
j = np.array((c1,c2))
print(j)
O = np.linalg.inv(i)@j
print(O)

r = np.linalg.norm(O-A)
print(r)
area = np.pi*(r**2)
print(area)

n = 64#no. of sides
t = np.linspace(0, 2*np.pi, n+1) #angle(start point, end point, one side greater than 'n')
x = O[0] + r*np.cos(t)
y = O[1] + r*np.sin(t)

plt.plot(O[0], O[1],'o')
plt.text(O[0], O[1],"O")
plt.axis('equal')
plt.grid()
plt.plot(x,y)
plt.show()