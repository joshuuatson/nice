import scipy.io as sio
from pylab import *
import numpy as np

#this takes in two signals and a time interval dt
#each signal is a 2D array with shape (T, N)
#where T is number of trials, and N is number of time points per trial
#so it has the N data points from each T trials


def field_field_coherence(x,y,dt, trl_indx = None):
    print(x.shape)
    if trl_indx is not None:
        x = x[trl_indx,:]
        y = y[trl_indx,:]
    print(x.shape)
        
    T = x.shape[0]    
    N = x.shape[1] 
    
    SYY = np.zeros(int(N/2+1))                             # Variable to store field spectrum.
    SXX = np.zeros(int(N/2+1))                             # Variable to store spike spectrum.
    SYX = np.zeros(int(N/2+1), dtype=complex)              # Variable to store cross spectrum.

    for k in range(T):                                 # For each trial

        yf = rfft((y[k,:]-np.mean(y[k,:])) *hanning(N))   #subtracts the mean and applies hanning window
        nf = rfft((x[k,:]-np.mean(x[k,:])) *hanning(N))   
        SYY = SYY + ( real( yf*conj(yf) ) )/T           # Field spectrum
        SXX = SXX + ( real( nf*conj(nf) ) )/T           # Field spectrum power spectrum densities
        SYX = SYX + (          yf*conj(nf)   )/T        # Cross spectrum

    coh = abs(SYX) / np.sqrt(SYY) / np.sqrt(SXX)             # Spike-field coherence
    f = rfftfreq(N, dt)   #frequency vector associated with FT
    
    return coh, f   

def spike_field_coherence(x,y,dt, trl_indx = None):
    
    if trl_indx is not None:
        x = x[trl_indx,:]
        y = y[trl_indx,:]
        
    T = x.shape[0]    
    N = x.shape[1] 
    
    SYY = np.zeros(int(N/2+1))                             # Variable to store field spectrum.
    SXX = np.zeros(int(N/2+1))                             # Variable to store spike spectrum.
    SYX = np.zeros(int(N/2+1), dtype=complex)              # Variable to store cross spectrum.

    for k in range(T):                                 # For each trial

        yf = rfft((y[k,:]-np.mean(y[k,:])) *hanning(N))   
        nf = rfft((x[k,:]-np.mean(x[k,:])))   
        SYY = SYY + ( real( yf*conj(yf) ) )/T           # Field spectrum
        SXX = SXX + ( real( nf*conj(nf) ) )/T           # Field spectrum
        SYX = SYX + (          yf*conj(nf)   )/T        # Cross spectrum

    coh = abs(SYX) / np.sqrt(SYY) / np.sqrt(SXX)             # Spike-field coherence
    f = rfftfreq(N, dt) 
    
    return coh, f