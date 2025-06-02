from pylab import *
import numpy as np

def field_field_coherence(x,y,dt, trl_indx = None):
    '''
    This takes in two signals and a time interval dt, where each
    signal is a 2D array with shape (T, N), T being number of trials,
    N being number of time points per trial. 

    If trl_indx is given, the funciton will select only those trials from both signals

    SYY and SXX hold the power spectra of signal Y and X respectively
    SYX holds the cross spectrum between Y and X

    NB that each is allocated an array of length int(N/2 +1), which is the number of 
    frequency bins obtained from the real FFT

    For each trial, the mean is subtracted from the signal, and a Hanning window is applied, 
    this windowing helps to reduce edge effects when performing the FFT

    Power spectra is calculated by taking the average across trials of the real part of the 
    signal * complec conjugate of the signal. (SXY via FFT of y * conj FFT of x)

    coh(f) = abs(SYX) / sqrt(SYY)*sqrt(SXX)

    the function also returns a frequency vector, computed using rfftfreq(N, dt). 
    '''
    if trl_indx is not None:
        x = x[trl_indx,:]
        y = y[trl_indx,:]
#    print(x.shape)
        
    T = x.shape[0]    
    N = x.shape[1] 
    
    #generating arrays
    SYY = np.zeros(int(N/2+1))                            
    SXX = np.zeros(int(N/2+1))                        
    SYX = np.zeros(int(N/2+1), dtype=complex)           

    #loop over trials
    for k in range(T):                                

        yf = rfft((y[k,:]-np.mean(y[k,:])) *hanning(N))   #rfft of y 
        nf = rfft((x[k,:]-np.mean(x[k,:])) *hanning(N))   #rfft of x
        SYY = SYY + ( real( yf*conj(yf) ) )/T       #this appends power spectra (for y) for this trial
        SXX = SXX + ( real( nf*conj(nf) ) )/T           # ""  for x
        SYX = SYX + ( yf*conj(nf)   )/T     # "" xy

    coh = abs(SYX) / np.sqrt(SYY) / np.sqrt(SXX)            
    f = rfftfreq(N, dt)
    
    return coh, f   



def spike_field_coherence(x,y,dt, trl_indx = None):
    '''
    This function expects a continuous signal (y) and a spike train (x)
    both should be shape (T, N) as above, x shouuld be preprocessed into a binned time series
    e.g. counts per time bin
    '''
    
    if trl_indx is not None:
        x = x[trl_indx,:]
        y = y[trl_indx,:]
        
    T = x.shape[0]    
    N = x.shape[1] 
    
    SYY = np.zeros(int(N/2+1))                             
    SXX = np.zeros(int(N/2+1))                             
    SYX = np.zeros(int(N/2+1), dtype=complex)              

    for k in range(T):                               

        yf = rfft((y[k,:]-np.mean(y[k,:])) *hanning(N))   
        nf = rfft((x[k,:]-np.mean(x[k,:])))     #hanning not applied to spiking signal
        SYY = SYY + ( real( yf*conj(yf) ) )/T         
        SXX = SXX + ( real( nf*conj(nf) ) )/T         
        SYX = SYX + (  yf*conj(nf)   )/T        

    coh = abs(SYX) / np.sqrt(SYY) / np.sqrt(SXX)            
    f = rfftfreq(N, dt) 
    
    return coh, f