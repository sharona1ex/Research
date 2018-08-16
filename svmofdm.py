# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:05:51 2018

@author: SHALOM ALEXANDER
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix

K = 8
CP = K//4
#P = int (input('Enter the number of pilot bits(it should be the multiple of two):'))
P = 8
if P%2 != 0:
    P = P + 1
pilotValue = np.array([])
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
pilotNum = P//2
pilotCar = np.arange(pilotNum)
mu = 2
payloadBits_per_OFDM = K*mu

SNRdb = 25  # signal to noise-ratio in dB at the receiver
SNR = [5, 6, 8, 10, 12, 15, 20, 25, 30]
points = len(SNR)

def Modulation(bits):                                        
    bit_r = bits.reshape((int(len(bits)/mu), mu))                  
    return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1) # This is just for QAM modulation

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def channel(signal,channelResponse,SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal,signalType='message'):
    if signalType == 'pilot':
        return signal[CP:(CP+pilotNum)]
    elif signalType == 'message':
        return signal[CP:(CP+K)]
    

def ofdm_simulate(codeword, channelResponse,SNRdb):       
    OFDM_data = np.zeros(pilotNum, dtype=complex)
    OFDM_data[pilotCar] = pilotValue
    #print('ofdm_data :',len(OFDM_data))
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    #print('ofdm_withCP :',len(OFDM_withCP))
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse,SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX,'pilot')
    #print('pilotSize :',len(OFDM_RX_noCP))
    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse,SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    return np.concatenate((OFDM_RX_noCP,OFDM_RX_noCP_codeword)), abs(channelResponse) 
    #return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) 


map_table = {
(-1.,-1.):0,
(-1.,1.):1,
(1.,-1.):2,
(1.,1.):3        
}
demap_table = {
        0 : (0,0),
        1 : (0,1),
        2 : (1,0),
        3 : (1,1)
        }

def Mapping(bits):
    return np.array([map_table[tuple(b)] for b in bits])





pilot = np.random.binomial(n=1, p=0.5, size=(P, )) 
    
pilotValue = Modulation(pilot)

channel_response = np.array([0.3+0.3j,0,1])
bits1 = np.array([])
signal_output = np.array([])
clf = svm.SVC(kernel='linear',C=1.0)
for i in range(0,100):    
    bitset = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
    signal, para = ofdm_simulate(bitset,channel_response,SNRdb)   
    bitset = np.hstack([pilot,bitset])
    bits1 = np.hstack([bits1,bitset]) # 160bits + 80 pilots = 240 bits
    signal_output = np.hstack([signal_output,signal]) #80 (2bits) + 40 pilotValues = 120 (2bits) of data

#    bit processing
#    bits1 = bitset
#    bits1 = Modulation(bits1)
#    real = np.real(bits1)
#    img = np.imag(bits1)
#    bits1 = np.concatenate((real,img))
#    bits1 = bits1.reshape(len(bits1)//2,2)
#    Label = Mapping(bits1.tolist()) #120
#    
#    #signal processing
#    signal_output = signal
#    mag = np.abs(signal_output)
#    theta = np.angle(signal_output,deg = True)
#    myinput = np.array([mag,theta])
#    myinput = myinput.reshape(len(signal_output),2)
#    clf.fit(myinput,Label)

        
    
    
print(len(bits1),"and",len(signal_output))
    
#
#real1 = np.real(pilotValue)
#img1 = np.imag(pilotValue)
#pilotValue = np.concatenate((real1,img1))
#pilotValue = pilotValue.reshape(4,2)
#pilotLabel = Mapping(pilotValue.tolist())
#pilotLabel = pilotLabel.tolist()
#print(pilotValue)
#print(pilotLabel)
#
bits1 = Modulation(bits1)
real = np.real(bits1)
img = np.imag(bits1)
bits1 = np.concatenate((real,img))
#print(len(bits1))
#print(bits1)
#print(bits1.reshape(8,2))
bits1 = bits1.reshape(len(bits1)//2,2)
Label = Mapping(bits1.tolist()) #120
#Label = Label.tolist()
#Label = pilotLabel + Label
#Label = np.array(Label)
#print(Label)
#
mag = np.abs(signal_output)
#print(mag)
theta = np.angle(signal_output,deg = True)
#print(theta)
#
#plt.scatter(theta,mag)
#plt.show()
#
#
myinput = np.array([mag,theta])
myinput = myinput.reshape(len(signal_output),2)
#print(myinput)
#
clf = svm.SVC(kernel='linear',C=1.0)
clf.fit(myinput,Label)
#
#
#
p = clf.predict(myinput[0:100])
result_1 = confusion_matrix(p,Label[0:100])
print(result_1)
#
#
#
