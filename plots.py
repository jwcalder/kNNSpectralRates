import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 12})
styles = ['^-','o-','d-','s-','pm-','xc-','*y-']



m=2
D = pd.read_csv('error.csv',header=None).values[:,:-1]
n = D[:,1].astype(int)
n_uni = np.unique(n)
Davg = np.zeros((len(n_uni),D.shape[1]))
i = 0
for nval in n_uni:
    Davg[i,:] = np.mean(D[n==nval,:],axis=0)
    i+=1

n = Davg[:,1].astype(int)
k = Davg[:,2].astype(int)
eps = Davg[:,3].astype(float)
val_err_knn = Davg[:,4:4+9]
val_err_eps = Davg[:,13:13+9]
vec_err_knn = np.sqrt(1-Davg[:,22:22+9])
vec_err_eps = np.sqrt(1-Davg[:,31:])

plt.figure()
kn = (k/n)**(1/m)
plt.loglog(kn,np.mean(val_err_knn[:,1:4],axis=1),styles[0],label=r'$\lambda_{1,2,3}$')
print(np.polyfit(np.log(kn),np.log(np.mean(val_err_knn[:,1:4],axis=1)),1))
plt.loglog(kn,np.mean(val_err_knn[:,4:],axis=1),styles[1],label=r'$\lambda_{4,5,6,7,8}$')
print(np.polyfit(np.log(kn),np.log(np.mean(val_err_knn[:,4:],axis=1)),1))
plt.loglog(kn,np.mean(vec_err_knn[:,1:4],axis=1),styles[2],label=r'$v_{1,2,3}$')
print(np.polyfit(np.log(kn),np.log(np.mean(vec_err_knn[:,1:4],axis=1)),1))
plt.loglog(kn,np.mean(vec_err_knn[:,4:],axis=1),styles[3],label=r'$v_{4,5,6,7,8}$')
print(np.polyfit(np.log(kn),np.log(np.mean(vec_err_knn[:,4:],axis=1)),1))
plt.xlim(1.02*np.max(kn),0.98*np.min(kn))
plt.xlabel(r'$(k/n)^{1/m}$',fontsize=18)
plt.ylabel('$L^2$ error',fontsize=18)
plt.legend(loc='upper right',fontsize=16)
plt.tight_layout()
ax = plt.gca()
ax.grid(which='both', axis='both', linestyle='--')
plt.savefig('knn_plot.pdf')

plt.figure()
plt.loglog(eps,np.mean(val_err_eps[:,1:4],axis=1),styles[0],label=r'$\lambda_{1,2,3}$')
print(np.polyfit(np.log(kn),np.log(np.mean(val_err_eps[:,1:4],axis=1)),1))
plt.loglog(eps,np.mean(val_err_eps[:,4:],axis=1),styles[1],label=r'$\lambda_{4,5,6,7,8}$')
print(np.polyfit(np.log(kn),np.log(np.mean(val_err_eps[:,4:],axis=1)),1))
plt.loglog(eps,np.mean(vec_err_eps[:,1:4],axis=1),styles[2],label=r'$v_{1,2,3}$')
print(np.polyfit(np.log(kn),np.log(np.mean(vec_err_eps[:,1:4],axis=1)),1))
plt.loglog(eps,np.mean(vec_err_eps[:,4:],axis=1),styles[3],label=r'$v_{4,5,6,7,8}$')
print(np.polyfit(np.log(kn),np.log(np.mean(vec_err_eps[:,4:],axis=1)),1))
plt.xlim(1.02*np.max(eps),0.98*np.min(eps))
plt.xlabel(r'$\varepsilon$',fontsize=18)
plt.ylabel('$L^2$ error',fontsize=18)
plt.legend(loc='upper right',fontsize=16)
plt.tight_layout()
ax = plt.gca()
ax.grid(which='both', axis='both', linestyle='--')
plt.savefig('eps_plot.pdf')

plt.show()
