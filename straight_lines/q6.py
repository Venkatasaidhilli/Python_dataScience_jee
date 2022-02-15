import numpy as np
import matplotlib.pyplot as plt
from coeffs import *

#Given
A = np.array([0,0])
B = np.array([2,0])
C = np.array([2,2])
D = np.array([0,2])
t = -np.pi/6
r1 = np.array([np.cos(t),-np.sin(t)])
r2 = np.array([np.sin(t),np.cos(t)])
T = (r1,r2)
P = A@T
Q = B@T
R = C@T
S = D@T
e1 = np.array([1,0])
x_sum = (A + B + C + D)@e1
x_sum_T = (P + Q + R + S)@e1

print("Sum of abscisses of the vertices of the square is:",x_sum)
print("Sum of abscisses of the vertices of the rotated square is:",x_sum_T)

#ploting lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)
x_PQ = line_gen(P,Q)
x_QR = line_gen(Q,R)
x_RS = line_gen(R,S)
x_SP = line_gen(S,P)
plt.plot(x_AB[0,:],x_AB[1,:],label="$AB$")
plt.plot(x_BC[0,:],x_BC[1,:],label="$BC$")
plt.plot(x_CD[0,:],x_CD[1,:],label="$CD$")
plt.plot(x_DA[0,:],x_DA[1,:],label="$DA$")
plt.plot(x_PQ[0,:],x_PQ[1,:],label="$PQ$")
plt.plot(x_QR[0,:],x_QR[1,:],label="$QR$")
plt.plot(x_RS[0,:],x_RS[1,:],label="$RS$")
plt.plot(x_SP[0,:],x_SP[1,:],label="$SP$")

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

plt.plot(Q[0],Q[1],'o')
plt.plot(R[0],R[1],'o')
plt.text(Q[0]*(1.02),Q[1]*(1.02), 'Q')
plt.text(R[0]*(1.02),R[1]*(1.02),'R')
plt.plot(S[0],S[1],'o')
plt.text(S[0]*(1.01),S[1]*(1.01),'S')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc = 'best')
plt.grid()
plt.axis('equal')

plt.show()


