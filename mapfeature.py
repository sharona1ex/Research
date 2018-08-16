# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 21:53:25 2018

@author: SHALOM ALEXANDER
"""
import numpy as np
import ofdm_system as ofdm

def mapfeature(x1,x2,n):
    degree = n
    out = np.zeros(len(x1))
    #out = out.reshape(4,3)
    for i in range(1,degree+1):
        for j in range(0,i+1):
            out = np.vstack((out,x1**(i-j)*x2**(j)))
    return out[1:].transpose()


#ofdm parameters
qam = 4
K = 3
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

#ofdm system creation
system1 = ofdm.Ofdm(K,P,qam,complex_map,mod_map)
print('ofdm system created.')

#data generation
[Feature_train,Label_train,Feature_test,Label_test,Feature_cv,Label_cv] = system1.dataGen(m,channel_response,SNRdb)
print('ofdm data created.')

#Feature data integration
Feature = np.vstack((Feature_train,Feature_test,Feature_cv))
Label = np.vstack((Label_train,Label_test,Label_cv))

np.save('Labelk3.npy',Label)
np.save('Featurek3.npy',Feature)
print('Feature data saved to Featurek3.npy & Labelk3.npy.')
print('Feature data integrated.')

#Feature data segregation
Feature_pilot = Feature[:,[0,1]]
Feature_mag = Feature[:,[2,3,4]]
Feature_ang = Feature[:,[5,6,7]]
print('Feature data segregrated.')

#polynomial formation
new_Feature = Feature_pilot
poly_Feature = []
print('Applying polynomial features...')
d = [2,3]
for degree in d:
    new_Feature = Feature_pilot
    for i in range(K):
        print('.',end='')
        x1 = Feature_mag[:,i]
        x2 = Feature_ang[:,i]
        Fnew = mapfeature(x1,x2,degree)
        new_Feature = np.hstack((new_Feature,Fnew))        
    poly_Feature.append(new_Feature)
    print('degree = %d completed.'% degree)


print('Polynomial Features Created.')

for i in range(len(d)):
    name = 'poly_Featurek' + str(K) + 'd'  + str(d[i]) + '.npy'
    np.save(name,poly_Feature[i])

print('Polynomial Features Saved.')







