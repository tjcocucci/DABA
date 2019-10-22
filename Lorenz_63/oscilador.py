#%%

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from tqdm import tqdm
#%%

ntimes = 1000
dt = 0.01
time = np.arange(0, ntimes*dt, dt)

true_state = np.zeros((ntimes,2))
true_state[0, :] = np.array([0.0,1.0])

std_error_obs = 2

omega_sq = 2.0
M = np.array([
        [ -np.power(dt, 2)*omega_sq + 1.0, dt ],
        [ -dt*omega_sq, 1.0 ]
        ])

#%%
Qtrue = np.eye(2)

# Generate truth and observations
for it in range(1, ntimes)  :
    
    true_state[it, :] = np.dot(M, true_state[it-1,:])    
    true_state[it, :] += st.multivariate_normal(cov=Qtrue).rvs() 
    
    
Rtrue = np.eye(2)*std_error_obs**2 
 
observacion = true_state + st.multivariate_normal(cov=Rtrue).rvs(ntimes)

#%%
    
plt.figure()    
plt.plot( time , true_state[:,0] )
plt.xlabel('X')
plt.ylabel('Tiempo')
plt.title('X en funcion del tiempo')
plt.savefig('ejemplo1.png')                #Guardo la figura en un archivo.

#Generar observaciones de la variable X agregando ruido a la secuencia verdadera.

plt.figure()
plt.plot(time , observacion[:, 0] - true_state[:, 0])
plt.xlabel('Error')
plt.ylabel('Tiempo')
plt.title('Error de la observacion en funcion del tiempo')
plt.savefig('ejemplo2.png')                #Guardo la figura en un archivo.


plt.figure()
plt.plot( time , observacion[:, 0] , 'ok')
plt.plot( time , true_state[:, 0] , '-b')
plt.xlabel('Error')
plt.ylabel('Tiempo')
plt.title('Observacion y X en funcion del tiempo')
plt.savefig('ejemplo3.png')                #Guardo la figura en un archivo.

#%%

def OI_analysis(obs, x0, M, B, H, R, Q=None):
    nx = len(x0)
    ntimes = obs.shape[0]
    
    xf = np.zeros((ntimes, nx))
    xa = np.zeros((ntimes, nx))
    xf[0, :] = x0
    xa[0, :] = x0

    for i in tqdm(range(1, ntimes)):
        # Forecast
        xf[i, :] = np.dot(M, xa[i-1, :])
        if Q is not None:
            xf[i, :] += st.multivariate_normal(cov=Q).rvs()
            
        #Analysis
        innovation = obs[i, :] - H.dot(xf[i, :])
        gain = B.dot(H.T).dot(np.linalg.inv(R + H.T.dot(B.dot(H))))
        xa[i, :] = xf[i, :] + np.dot(gain, innovation)
    
    return xf, xa

#%%

x0 = np.array([2, 1])
# B = np.cov(observacion.T) * 2
B = np.eye(2) + Qtrue
H = np.eye(2)
R = np.copy(Rtrue)

xf, xa = OI_analysis(observacion, x0, M, B, H, R)

#%%

plt.plot(time, observacion[:, 0], 'r.')
plt.plot(time, xa[:, 0])
plt.plot(time, true_state[:, 0], 'k--')

#%%
RMSE_analysis = np.sqrt(np.mean((true_state - xa)**2, axis=1))
plt.plot(RMSE_analysis)

