import numpy as np
import matplotlib.pyplot as plt

P = np.array((4,7))
O = np.array((0,0))
r = np.sqrt(9)
print(r)
a=1
b=2*P
c=(np.linalg.norm(P)**2)-9
print('Product of roots =', c)
#print(c)

n = 200
t = np.linspace(0,2*np.pi, n+1)
x = O[0] + r*np.cos(t)
y = O[1] + r*np.sin(t)

plt.plot(O[0],O[1],'o')
plt.text(O[0],O[1], 'O')
plt.plot(P[0],P[1],'o')
plt.text(P[0],P[1], 'P')
plt.plot(x,y)
plt.axis('equal')
plt.grid()
plt.show()