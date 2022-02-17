import numpy as np
import matplotlib.pyplot as plt
from coeffs import *

A = np.array((5,-2))
B = np.array((-2,5))

def line_gen(A,B):
    len = 10
    x_AB = np.zeros((2,len))
    lam_1 = np.linspace(0,1,len)
    
    for i in range(len):
        temp1 = A + lam_1[i]*(B-A)
        x_AB[:,i] = temp1.T
    return x_AB

O1 = np.array((1,0))
n = np.array((1,1))
m = omat@n
print(m)

c1 = 3
r1 = (np.linalg.norm(O1))**2
print(r1)

c2 = (m.T)@O1
print(c2)

i = np.array((n,m))
j = np.array((c1,c2))
P = np.linalg.inv(i)@j
print(P)

O2 = 2*P-O1
print(O2)

n = 200
t = np.linspace(0, 2*np.pi, n+1)
x1 = O1[0] + r1*np.cos(t)
y1 = O1[1] + r1*np.sin(t)
x2 = O2[0] + r1*np.cos(t)
y2 = O2[1] + r1*np.sin(t)

def line_gen(O1,O2):
    len = 10
    x_O1O2 = np.zeros((2,len))
    lam_2 = np.linspace(0,1,len)
    
    for i in range(len):
        temp2 = O1 + lam_2[i]*(O2-O1)
        x_O1O2[:,i] = temp2.T
    return x_O1O2


# line gen
x_AB = line_gen(A,B)
x_O1O2 = line_gen(O1,O2)

#ploting
plt.plot(x_AB[0,:],x_AB[1,:],label= "$AB$")
plt.plot(x_O1O2[0,:],x_O1O2[1,:],label= "$O1O2$")

plt.plot(A[0],A[1], 'o')
plt.text(A[0],A[1],'A')
plt.plot(B[0],B[1],'o')
plt.text(B[0],B[1],'B')

plt.plot(O1[0], O1[1], 'o')
plt.text(O1[0],O1[1], 'O1')
plt.plot(O2[0], O2[1], 'o')
plt.text(O2[0],O2[1], 'O2')
plt.plot(P[0], P[1], 'o')
plt.text(P[0],P[1], 'P')

plt.plot(x1,y1)
plt.plot(x2,y2)

plt.grid()
plt.axis('equal')
plt.show()