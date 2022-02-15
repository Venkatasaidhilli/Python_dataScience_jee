import numpy as np
import matplotlib.pyplot as plt
from coeffs import *

def ortho(k):
    X = np.zeros(2)
    A = np.array([k,-3*k])
    B = np.array([5,k])
    C = np.array([-k,2])
    m1 = dir_vec(B,C)
    m2 = dir_vec(A,C)
    Aug = (m1,m2)
    c1 = m1@A
    c2 = m2@B
    cs = np.array([c1,c2])
    X = np.linalg.inv(Aug)@cs
    return (X)
def A(k):
    A = np.array([k,-3*k])
    return(A)
def B(k):
    B = np.array([5,k])
    return(B)
def C(k):
    C = np.array([-k,2])
    return(C)

    
 # 5K2 + 13K - 46
#FROM THE solved EQUATION
a = 5
b = 13
c = -46
k1 = np.sqrt(b**2 - 4*a*c) - b
k1 =  k1/(2*a) 
K2 = np.sqrt(b**2 - 4*a*c) + b
k2 = - K2/(2*a)
print(k1)
print(k2)       
X1 = ortho(k1)
X2 = ortho(k2)
print (X1)
print (X2)
A1 = A(k1)
B1 = B(k1)
C1 = C(k1)

A2 = A(k2)
B2 = B(k2)
C2 = C(k2)

x_AB = line_gen(A1,B1)
x_BC = line_gen(B1,C1)
x_AC = line_gen(A1,C1)

x_XY = line_gen(A2,B2)
x_YZ = line_gen(B2,C2)
x_ZX = line_gen(A2,C2)
#Triangle with k1
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_AC[0,:],x_AC[1,:],label='$AC$')
plt.plot(A1[0], A1[1], 'o')
plt.text(A1[0] * (1 + 0.1), A1[1] * (1 - 0.1) , 'A1')
plt.plot(B1[0], B1[1], 'o')
plt.text(B1[0] * (1 - 0.2), B1[1] * (1) , 'B1')
plt.plot(C1[0], C1[1], 'o')
plt.text(C1[0] * (1 - 0.2), C1[1] * (1) , 'C1')
#Triangle with k2
plt.plot(x_XY[0,:],x_XY[1,:],label='$XY$')
plt.plot(x_YZ[0,:],x_YZ[1,:],label='$YZ$')
plt.plot(x_ZX[0,:],x_ZX[1,:],label='$ZX$')
plt.plot(A2[0], A2[1], 'o')
plt.text(A2[0] * (1 + 0.1), A2[1] * (1 - 0.1) , 'A2')
plt.plot(B2[0], B2[1], 'o')
plt.text(B2[0] * (1 - 0.2), B2[1] * (1) , 'B2')
plt.plot(C2[0], C2[1], 'o')
plt.text(C2[0] * (1 - 0.2), C2[1] * (1) , 'C2')
#Orthocenters
plt.plot(X1[0], X1[1], 'o')
plt.text(X1[0] * (1 + 0.1), X1[1] * (1 - 0.1) , 'X1')
plt.plot(X2[0], X2[1], 'o')
plt.text(X2[0] * (1 - 0.2), X2[1] * (1) , 'X2')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.axis('equal')
plt.grid()
plt.show()
