import numpy as np
import matplotlib.pyplot as plt


# end points
A = np.array([0 ,-5/6])
B = np.array([-5/8, 0])
C = np.array([0,10/3])
D = np.array([5/2,0])
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
x_CD = line_gen(C,D)

#ploting
plt.plot(x_AB[0,:],x_AB[1,:],label= "$AB$")
plt.plot(x_CD[0,:],x_CD[1,:],label = "$CD$")

plt.plot(A[0],A[1], 'o')
plt.text(A[1]*(1.1),A[1]*(1.3),'A')
plt.plot(B[0],B[1],'o')
plt.text(B[0]*(1.1),B[1]*(1.2),'B')

plt.plot(C[0],C[1],'o')
plt.plot(D[0],D[1],'o')
plt.text(C[0]*(1.2),C[1]*(1.2), 'C')
plt.text(D[0]*(1.2),D[1]*(1.2),'D')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc = 'best')
plt.grid()

plt.show()

