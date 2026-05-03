import numpy as np
from scipy.special import erfc
from mittag_leffler import ml,I_Rabotnov
import matplotlib.pyplot as plt

z = np.linspace(-2., 2., 50)
assert np.allclose(ml(z, 1.), np.exp(z))

z = np.linspace(-2., 2., 50)
assert np.allclose(ml(z**2, 2.), np.cosh(z))

z = np.linspace(0., 2., 50)
assert np.allclose(ml(np.sqrt(z), 0.5), np.exp(z)*erfc(-np.sqrt(z)))

# z = np.linspace(-50., 10., 1000)
# for alpha in [0.00001]+range(1,6):
#     plt.plot(z,ml(z, alpha))
# 
# plt.grid(True,which='both')
# plt.axis([-50.,10.,-3.,4.])
# plt.show()


T=np.logspace(-6., 2., 1000)
for beta in [0.1,1.,3.]:
    for alpha in np.linspace(-1.+1.e-10,0.,6):
        plt.plot(T,I_Rabotnov(T, alpha, beta))
plt.grid(True,which='both')
plt.show()

# T=np.logspace(-6., 2., 1000)
mu0=1.7 ; alpha0=-0.46 ; beta0=0.98 ; lambda0=-0.495
plt.plot(T,mu0*(1.+lambda0*I_Rabotnov(T, alpha0, beta0)))
plt.plot(T,1./mu0*(1.-lambda0*I_Rabotnov(T, alpha0, beta0+lambda0)))
plt.grid(True,which='both')
plt.show()

# Nyquist
f=lambda p:1./(mu0*(1.+lambda0/(p**(1+alpha0)+beta0)))
tomega=np.logspace(-10.,10.,1000)
X=[];Y=[]
for omega in tomega:
    Z=f(omega*1.j)
    X.append(Z.real)
    Y.append(-Z.imag)
plt.plot(X,Y,'+')
plt.grid(True,which='both')
plt.axis('equal')
plt.show()
