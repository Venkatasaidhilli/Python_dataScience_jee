import numpy as np
import matplotlib.pyplot as plt

r1 = 3
P = np.array((2,2))
O2 = np.array((-1,2))
r2 = np.sqrt((np.linalg.norm(O2))**2+4)
print(r2)
O1 = 2*P-O2
print(O1)
m = np.array((1,0))

a = 1
b = -10
c = 20
lam = (np.sqrt(b**2-4*a*c)-b)/2*a
print('The length of the intercept by circle C on x axis is = ', lam)

r = r1 = r2
n = 200
t = np.linspace(0, 2*np.pi, n+1)
x1 = O1[0] + r*np.cos(t)
y1 = O1[1] + r*np.sin(t)
x2 = O2[0] + r*np.cos(t)
y2 = O2[1] + r*np.sin(t)

plt.plot(O1[0], O1[1], 'o')
plt.text(O1[0],O1[1], 'O1')
plt.plot(O2[0], O2[1], 'o')
plt.text(O2[0],O2[1], 'O2')
plt.plot(P[0], P[1], 'o')
plt.text(P[0],P[1], 'P')

plt.plot(x1,y1)
plt.plot(x2,y2)
plt.axis('equal')
plt.grid()
plt.show()