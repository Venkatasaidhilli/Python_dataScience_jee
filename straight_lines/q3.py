import numpy as np
import matplotlib.pyplot as plt
from coeffs import *

def line_gen(A,B):
    len = 10
    x_AB = np.zeros((2,len))
    lam_1 = np.linspace(0,1,len)
    
    for i in range(len):
        temp1 = A + lam_1[i]*(B-A)
        x_AB[:,i] = temp1.T
    return x_AB
#GIVEN
n1 =  np.array([1,-1])#AB
n2 = np.array([7,-1])#AD
c1 = -1#AB
c2 = 5#AD
P = np.array([-1,-2])
#SOLUTION
i = (n1,n2)
a = np.array([c1,c2])
A = np.linalg.inv(i)
A = A@a
print (A)
C = 2*P - A
print(C)
c3 = n2@C#BC
c4 = n1@C#CD
#solving AB and BC
b = np.array([c1,c3])
B = np.linalg.inv(i)@b
print(B)
#solving CD and AD
d = np.array([c4,c2])
D = np.linalg.inv(i)@d
print(D)

#ploting lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)
x_AC = line_gen(A,C)
x_BD = line_gen(B,D)

plt.plot(x_AB[0,:],x_AB[1,:],label="$AB$")
plt.plot(x_BC[0,:],x_BC[1,:],label="$BC$")
plt.plot(x_CD[0,:],x_CD[1,:],label="$CD$")
plt.plot(x_DA[0,:],x_DA[1,:],label="$DA$")
plt.plot(x_AC[0,:],x_AC[1,:],label="$AC$")
plt.plot(x_BD[0,:],x_BD[1,:],label="$BD$")

plt.plot(A[0],A[1], 'o')
plt.text(A[1]*(1.01),A[1]*(1.01),'A')
plt.plot(B[0],B[1],'o')
plt.text(B[0]*(1.01),B[1]*(1.02),'B')
plt.plot(P[0],P[1],'o')
plt.text(P[0]*(1.01),P[1]*(1.01),'P')
plt.plot(C[0],C[1],'o')
plt.plot(D[0],D[1],'o')
plt.text(C[0]*(1.02),C[1]*(1.02), 'C')
plt.text(D[0]*(1.02),D[1]*(1.02),'D')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc = 'best')
plt.grid()
plt.axis('equal')

plt.show()






