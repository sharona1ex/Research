# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 22:26:06 2018

@author: SHALOM ALEXANDER
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:05:51 2018

@author: SHALOM ALEXANDER
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix

K = 1
CP = K//4
#P = int (input('Enter the number of pilot bits(it should be the multiple of two):'))
P = 2
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

map_table = {
(-1.,-1.):0,
(-1.,1.):1,
(1.,-1.):2,
(1.,1.):3        
}

def Mapping(bits):
    return np.array([map_table[tuple(b)] for b in bits])

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
    return OFDM_RX_noCP,OFDM_RX_noCP_codeword,abs(channelResponse) 



pilot = np.random.binomial(n=1, p=0.5, size=(P, ))
pilotValue = Modulation(pilot)

#channel_response = np.array([0.3+0.3j,0,1])
channel_response = np.array([[0.3+0.3j,0,1],[1, 0, 0.3+0.3j],[4, 0, 0.2+6j]])

dist_pilot = np.zeros(P//2)
bits1 = np.zeros(K*mu)
signal_output = np.zeros(K)
clf = svm.SVC(kernel = 'rbf',gamma = 0.01,C=5)
num = 5000
#channel_response_set_test2[np.random.randint(0,len(channel_response_set_test2))]
for i in range(0,num):    
    bitset = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
    dpilot,signal,para = ofdm_simulate(bitset,channel_response[np.random.randint(0,len(channel_response))],SNRdb)   
    dist_pilot = np.vstack((dist_pilot,dpilot))
#    dist_pilot = np.vstack((dist_pilot,pilotValue - dpilot))
    bits1 = np.vstack((bits1,bitset)) # 160bits + 80 pilots = 240 bits
    signal_output = np.vstack((signal_output,signal)) #80 (2bits) + 40 pilotValues = 120 (2bits) of data


Final_label = np.zeros(K)

#starting from 1 to skip the zeros added at the zeroth index
for i in range(1,num+1):    
    Label = Modulation(bits1[i])
    img = np.imag(Label)
    real = np.real(Label)
    n = len(real)
    Label = np.array([ [real[j],img[j]] for j in range(n)])
    Label = Mapping(Label.tolist())
    print(np.shape(Label),np.shape(Final_label))
    Final_label = np.vstack((Final_label,Label))
    
    
pilot_mag = np.abs(dist_pilot[1:]) 
pilot_angle = np.angle(dist_pilot[1:])

print(np.shape(pilot_mag)," ",np.shape(pilot_angle)," ",np.shape(bits1[1:])," ",np.shape(signal_output[1:]))
#Features = np.array([dist_pilot[1:],np.abs(signal_output[1:]),np.angle(signal_output[1:])])
#Feature = Features.reshape(len(signal_output),3)

#Features
pilot = pilot.tolist()
pilot_mag = pilot_mag.tolist()
pilot_angle = pilot_angle.tolist()
bit_mag = np.abs(signal_output[1:]).tolist()
bit_angle = np.angle(signal_output[1:]).tolist()
print(len(pilot + pilot_mag[0] + pilot_angle[0] + bit_mag[0] + bit_angle[0]))
Features = np.zeros(P + K*mu)    

for i in range(num):
    Features = np.vstack((Features,pilot_mag[i] + pilot_angle[i] + bit_mag[i] + bit_angle[i]))


#***************************************************    
print(np.shape(Features[1:]))
Features = Features[1:]
Final_label = Final_label[1:]
print(np.shape(Final_label))
#***************************************************

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


#to convert the 16 bit number to single classess
#use permutations for this

clf.fit(Feature_train,Label_train.tolist())
pred = clf.predict(Feature_test)
result_1 = confusion_matrix(Label_test,pred)
[r,c] = np.shape(result_1)
correct_sum = 0
for i in range(r):
    correct_sum += result_1[i][i]

number_of_examples = len(Feature_test)
error_sum = number_of_examples - correct_sum
average_test_error = error_sum/number_of_examples
print("Average test error:",average_test_error)
print(result_1)




