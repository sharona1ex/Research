# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 11:57:58 2018

@author: SHALOM ALEXANDER
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import svm

K = 4
CP = K//4
P = int (input('Enter the number of pilot bits(it should be the multiple of two):'))
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
    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP),np.imag(OFDM_RX_noCP))), np.concatenate((np.real(OFDM_RX_noCP_codeword),np.imag(OFDM_RX_noCP_codeword))))), abs(channelResponse) 

pilot = np.random.binomial(n=1, p=0.5, size=(P, )) 
    
pilotValue = Modulation(pilot)
clf = svm.SVC(kernel='rbf',gamma = 0.01,C=5.0)
### =================== Machine Learning Training ================
def BigTest(SNRdb):
    print('*******Big Test******* SNRdb=%d' % SNRdb)
    amount = 1000
    pred = np.array([])
    pred = np.arange(0) #just initializing numpy array
    signalop = np.arange(0) #just initializing numpy array
    channel_response_set_test2 = np.array([[0.3+0.3j,0,1],[1, 0, 0.3+0.3j]])
    bit_test2 = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM*amount, ))
    for i in range(0,2*K*(amount),2*K):
        signal_test2,_1 = ofdm_simulate(bit_test2[i:i+2*K],channel_response_set_test2[np.random.randint(0,len(channel_response_set_test2))],SNRdb)
        signalop = np.hstack((signalop,signal_test2))
    for i in range(0,len(signalop),2*K + 2*pilotNum):
        input_sample2 = []
        input_sample2.append(signalop[i:i+2*K + 2*pilotNum])
        predOct = clf.predict(input_sample2)
        predOctuint8 = np.array(predOct,dtype = np.uint8)
        predBin = np.unpackbits(predOctuint8,axis=-1)
        pred = np.hstack([predBin,pred])        
    #pred = np.packbits(predOct,axis=-1)
    print('Confusion Matrix')
    print('Size of pred array:',len(pred))
    print('Size of bits transmitted',len(bit_test2))
    result_1 = confusion_matrix(pred,bit_test2)
    print(result_1)
    print('BER:',(result_1[1,0] + result_1[0,1])/(result_1[0,0] + result_1[1,1]))
            
def training():     
        # The H information set
#   channel_response_set_train = np.array([[0.3+0.3j,0,1],[1, 0, 0.3+0.3j]])
#   channel_response_set_test = np.array([[0.3+0.3j,0,1],[1, 0, 0.3+0.3j]])
   channel_response_set_train = np.array([[0.3+0.3j,0,1]])
   channel_response_set_test = np.array([[0.3+0.3j,0,1]])
   print ('length of training channel response', len(channel_response_set_train), 'length of testing channel response', len(channel_response_set_test))
   total_batch = 1000      

   for index_m in range(total_batch):
       input_samples = []
       input_labels = []
       if index_m%100 == 0:
           print(index_m,"th/",total_batch," batch")
       for index_k in range(0, 100):
           bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
           channel_response = channel_response_set_train[np.random.randint(0,len(channel_response_set_train))]
           signal_output, para = ofdm_simulate(bits,channel_response,SNRdb)   
           input_labels.append(bits[0:2*K]) # prediciting 2K bits at a time
           input_samples.append(signal_output)  
       input_labels_in_octal = np.packbits(input_labels,axis=-1)
       batch_x = np.asarray(input_samples)
       batch_y = np.asarray(input_labels_in_octal)
       clf.fit(batch_x,batch_y)
   BigTest(SNRdb)
   
#*******************************************************************************
#let's test the function training()
training()
       
       
           