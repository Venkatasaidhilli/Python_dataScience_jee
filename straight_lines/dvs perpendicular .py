import numpy as np
import matplotlib.pyplot as plt
import math as mp
from coeffs import *


# end points
A = np.array([0 ,-4])
B = np.array([4, 0])


#C = np.array([2,(1-2*mp.sqrt(6))])
P = np.array([2,1])

m = np.array([1,1])
d = 2*mp.sqrt(3)

h = -d/np.linalg.norm(m)
Q = P + h*m
print (h)
print (Q)
print (A)
x= dir_vec(A,B)
y = norm_vec(A,B)
p=np.zeros(2)
p[0] = x@Q
p[1] = y@A
N = np.vstack((x,y))
C = np. linalg.inv(N)@p


def line_gen(A,B):
    len = 10
    x_AB = np.zeros((2,len))
    lam_1 = np.linspace(0,1,len)
    
    for i in range(len):
        temp1 = A + lam_1[i]*(B-A)
        x_AB[:,i] = temp1.T
    return x_AB

# line gen
x_AB = line_gen(A,B)
x_CD = line_gen(C,Q)


#ploting
plt.plot(x_AB[0,:],x_AB[1,:],label= "$AB$")
plt.plot(x_CD[0,:],x_CD[1,:],label = "$CQ$")

plt.plot(A[0],A[1], 'o')
plt.text(A[1]*(1),A[1]*(1),'A')
plt.plot(B[0],B[1],'o')
plt.text(B[0]*(1.1),B[1]*(1.2),'B')
plt.plot(P[0],P[1],'o')
plt.text(P[0]*(1.1),P[1]*(1.1),'P')
plt.plot(C[0],C[1],'o')
plt.plot(Q[0],Q[1],'o')
plt.text(C[0]*(1.2),C[1]*(1.2), 'C')
plt.text(Q[0]*(1.02),Q[1]*(1.02),'Q')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc = 'best')
plt.grid()
plt.axis('equal')

plt.show()

