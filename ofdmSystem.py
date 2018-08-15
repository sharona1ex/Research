# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:06:58 2018

@author: SHARON ALEXANDER
"""
import numpy as np
from math import log10
from sklearn import svm

class Ofdm:
    
    def __init__(self,K,P,qam,complex_map,mod_map):
        self.K = K
        self.P = P
        self.complex_map = complex_map
        self.mod_map = mod_map
        self.mu = int(log10(qam)/log10(2))
        self.CP = K//4
        self.pilotNum = P//self.mu
        self.pilotCar = np.arange(self.pilotNum)
        self.pilot = np.random.binomial(n=1, p=0.5, size=(self.P, ))
        self.pilotValue = self.modulation(self.pilot)
               
    def complexMap(self,z):
        #since z is complex seperate them
        #this function converts complex number into
        #set of integers to simplify calculations
        real = np.real(z)
        img = np.imag(z)
        comp = [[real[i],img[i]] for i in range(0,len(real))]
        return np.array([self.complex_map[tuple(b)] for b in comp])
    
    def modulation(self,bits):
        mu_bits = bits.reshape(len(bits)//self.mu,self.mu)
        return np.array([self.mod_map[tuple(b)]] for b in mu_bits.tolist())
    
    def addCP(self,OFDM_time):
        cp = OFDM_time[-self.CP:]               # take the last CP samples ...
        return np.hstack([cp, OFDM_time])  # ... and add them to the beginning
    
    def removeCP(self,signal,signalType='message'):
        if signalType == 'pilot':
            return signal[self.CP:(self.CP+self.pilotNum)]
        elif signalType == 'message':
            return signal[self.CP:(self.CP+self.K)]
        
    def ofdm_simulate(self,codeword, channelResponse,SNRdb):       
        OFDM_data = np.zeros(self.pilotNum, dtype=complex)
        print(self.pilotCar," pilotCar")
        print(self.pilotValue," pilotValue type: ",type(self.pilotValue))
        OFDM_data[self.pilotCar] = self.pilotValue.tolist()
        #print('ofdm_data :',len(OFDM_data))
        OFDM_time = self.IDFT(OFDM_data)
        OFDM_withCP = self.addCP(OFDM_time)
        #print('ofdm_withCP :',len(OFDM_withCP))
        OFDM_TX = OFDM_withCP
        OFDM_RX = self.channel(OFDM_TX,channelResponse,SNRdb)
        OFDM_RX_noCP = self.removeCP(OFDM_RX,'pilot')
        #print('pilotSize :',len(OFDM_RX_noCP))
        # ----- target inputs ---
        symbol = np.zeros(self.K, dtype=complex)
        codeword_qam = self.modulation(codeword)
        symbol[np.arange(self.K)] = codeword_qam
        OFDM_data_codeword = symbol
        OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
        OFDM_withCP_cordword = self.addCP(OFDM_time_codeword)
        OFDM_RX_codeword = self.channel(OFDM_withCP_cordword, channelResponse,SNRdb)
        OFDM_RX_noCP_codeword = self.removeCP(OFDM_RX_codeword)
        return OFDM_RX_noCP,OFDM_RX_noCP_codeword
    
    def dataGen(self,num,channel_response,SNRdb):
        dist_pilot = np.zeros(self.P//2)
        signal_output = np.zeros(self.K)
        payloadBits_per_OFDM = self.K*self.mu
        bits1 = np.zeros(payloadBits_per_OFDM)
        for i in range(0,num):    
            bitset = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
            dpilot,signal = self.ofdm_simulate(bitset,channel_response[np.random.randint(0,len(channel_response))],SNRdb)   
            dist_pilot = np.vstack((dist_pilot,dpilot))
            bits1 = np.vstack((bits1,bitset)) 
            signal_output = np.vstack((signal_output,signal)) 
        
        Final_label = np.zeros(self.K)
        #starting from 1 to skip the zeros added at the zeroth index
        for i in range(1,num+1):    
            Label = self.modulation(bits1[i])
            print(Label)
            Label = self.complex_map(Label)
            print(np.shape(Label),np.shape(Final_label))
            Final_label = np.vstack((Final_label,Label))
        
        pilot_mag = np.abs(dist_pilot[1:]) 
        pilot_angle = np.angle(dist_pilot[1:])
        print(np.shape(pilot_mag)," ",np.shape(pilot_angle)," ",np.shape(bits1[1:])," ",np.shape(signal_output[1:]))
        
        pilot_mag = pilot_mag.tolist()
        pilot_angle = pilot_angle.tolist()
        bit_mag = np.abs(signal_output[1:]).tolist()
        bit_angle = np.angle(signal_output[1:]).tolist()
        print(len(pilot_mag[0] + pilot_angle[0] + bit_mag[0] + bit_angle[0]))
        Features = np.zeros(self.P + payloadBits_per_OFDM)    
        
        for i in range(num):
            Features = np.vstack((Features,pilot_mag[i] + pilot_angle[i] + bit_mag[i] + bit_angle[i]))
            
        #***************************************************    
        print(np.shape(Features[1:]))
        Features = Features[1:]
        Final_label = Final_label[1:]
        print(np.shape(Final_label))
        #***************************************************
        #60:20:20 for train:test:cv
        #training set
        m = len(Features)
        Feature_train = Features[0:int(0.6*m)]
        Label_train = Final_label[0:int(0.6*m)]
        #test set
        mr = m - 0.6*m
        Feature_test = Features[int(0.6*m):int(0.6*m+mr*0.5)]
        Label_test = Final_label[int(0.6*m):int(0.6*m+mr*0.5)]
        #cross validation set
        Feature_cv = Features[int(0.6*m+mr*0.5):int(0.6*m+mr*0.5+mr*0.5)]
        Label_cv = Final_label[int(0.6*m+mr*0.5):int(0.6*m+mr*0.5+mr*0.5)]
        
        return Feature_train,Label_train,Feature_test,Label_test,Feature_cv,Label_cv

    def channel(self,signal,channelResponse,SNRdb):
        convolved = np.convolve(signal, channelResponse)
        signal_power = np.mean(abs(convolved**2))
        sigma2 = signal_power * 10**(-SNRdb/10)  
        noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
        return convolved + noise
    
    @staticmethod
    def IDFT(OFDM_data):
        return np.fft.ifft(OFDM_data)

    


#ofdm parameter setting
qam = 4
K = 1
CP = K//4
P = 2
if P%2 != 0:
    P = P + 1 

complex_map = {
        (-1.,-1.):0,
        (-1., 1.):1,
        ( 1.,-1.):2,
        ( 1., 1.):3    
        }

mod_map = {
        (0,0):-1.0 - 1.0j,
        (0,1):-1.0 + 1.0j,
        (1,0): 1.0 - 1.0j,
        (1,1): 1.0 + 1.0j
        }       

# channel settings
SNRdb = 25
channel_response = np.array([[0.3+0.3j,0,1],[1, 0, 0.3+0.3j],[4, 0, 0.2+6j]])

#Amount of data
m = 5000 # m mean number of examples

#Ofdm object and data creation
myOfdm = Ofdm(K,P,qam,complex_map,mod_map)
[Feature_train,Label_train,Feature_test,Label_test,Feature_cv,Label_cv] = myOfdm.dataGen(m,channel_response,SNRdb)

## SVM model
#clf = svm.SVC(kernel='rbf',gamma=0.01,C=5.0)
#clf.fit(Feature_train,Label_train)


    
    
    
    
    
    

    
    
        
        
    
