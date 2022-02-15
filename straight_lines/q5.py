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
O = np.array([0,0])    
n = np.array([1,1])
m = omat@n
i = (n,m)
c = 2
c1 = 0
j = np.array([2,0])
F = np.linalg.inv(i)@j
print(F)
P = -2*F
a = np.linalg.norm(P-F)
p = a*2/np.sqrt(3)
h = p/2
lam = h/np.linalg.norm(m)

Q = F - lam*m
R = F + lam*m
print(Q)
print(R)
x = np.linalg.norm(Q-R)
print(x)
print(p)
G = (P + Q)/2



#ploting lines
x_AB = line_gen(P,Q)
x_BC = line_gen(Q,R)
x_CD = line_gen(P,R)
x_DA = line_gen(P,F)
x_AC = line_gen(R,G)


plt.plot(x_AB[0,:],x_AB[1,:],label="$PQ$")
plt.plot(x_BC[0,:],x_BC[1,:],label="$QR$")
plt.plot(x_CD[0,:],x_CD[1,:],label="$PR$")
plt.plot(x_DA[0,:],x_DA[1,:],label="$PF$")
plt.plot(x_AC[0,:],x_AC[1,:],label="$RG$")


plt.plot(P[0],P[1], 'o')
plt.text(P[1]*(1.01),P[1]*(1.01),'P')
plt.plot(Q[0],Q[1],'o')
plt.text(Q[0]*(1.01),Q[1]*(1.02),'Q')
plt.plot(R[0],R[1],'o')
plt.text(R[0]*(1.01),R[1]*(1.01),'R')
plt.plot(F[0],F[1],'o')
plt.plot(G[0],G[1],'o')
plt.text(F[0]*(1.02),F[1]*(1.02), 'F')
plt.text(G[0]*(1.02),G[1]*(1.02),'G')
plt.plot(O[0],O[1], 'o')
plt.text(O[1]*(1.01),O[1]*(1.01),'O')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc = 'best')
plt.grid()
plt.axis('equal')

plt.show()




