
import numpy as np
import matplotlib.pyplot as plt


import subprocess
import shlex


def line_gen(A,B):
  len =10
  x_AB = np.zeros((2,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB


A = np.array([5/2,0]) 
B = np.array([0,10/3]) 

x_AB=line_gen(A,B)

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')

C = np.array([-5/8,0])
D = np.array([0,-5/6])
x_CD=line_gen(C,D)
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
plt.grid() 
plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 - 0.1), A[1] * (1 + 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 - 0.3), C[1] * (1 - 0.1) , 'C')
plt.plot(D[0], D[1], 'o')
plt.text(D[0] * (1 - 0.4), D[1] * (1 - 0.2) , 'D')
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.show()
