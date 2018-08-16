# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 20:48:07 2018

@author: SHALOM ALEXANDER
"""
import numpy as np
import multi_svm as ms
import matplotlib.pyplot as plt

def feature_split(Features):
    #60:20:20
    #train set
    m = len(Features)
    Feature_train = Features[0:int(0.6*m)]
    #test set
    mr = m - 0.6*m
    Feature_test = Features[int(0.6*m):int(0.6*m+mr*0.5)]
    #cross validation set
    Feature_cv = Features[int(0.6*m+mr*0.5):int(0.6*m+mr*0.5+mr*0.5)]
    return Feature_train,Feature_test,Feature_cv

Features = np.load('Featurek3.npy')
Label = np.load('Labelk3.npy')
print('Base Data loaded.')
[Feature_train,Feature_test,Feature_cv] = feature_split(Features)
[Label_train,Label_test,Label_cv] = feature_split(Label)
print('Base Data split into 60:20:20')

# d:number degree used K:number of carriers
#d = 2
K = 3
#
#poly_Feature = []
#
#for i  in range(d):
#    name = 'poly_Featurek' + str(K) + 'd' + str(d) + '.npy' 
#    poly_Feature.append(np.load(name))
#print('Polynomial Feature loaded.')


#SVM parameters
Kernel = 'rbf'
Gamma = 0.1
c = 7
clf = ms.multi_svm(K,Kernel,Gamma,c)
print('multi-svm models created')

train_error = []
test_error = []
print('Calculating learning curve for training set.')
for i in range(200,len(Feature_train),5):
    print('.',end="")
    clf.train(Feature_train[0:i,:],Label_train[0:i,:])
    [err,_] = clf.predict(Feature_train[0:i,:],Label_train[0:i,:])
    train_error.append(err)
print('\nCalculated learning curve for training set.')

print('Calculating learning curve for test set.')
for i in range(200,len(Feature_test),5):
    print('.',end="")
    clf.train(Feature_test[0:i,:],Label_test[0:i,:])
    [err,_] = clf.predict(Feature_test[0:i,:],Label_test[0:i,:])
    test_error.append(err)
print('\nCalculated learning curve for test set.')

train_num = [i for i in range(200,len(Feature_train),5)]  
test_num = [i for i in range(200,len(Feature_test),5)]

plt.plot(train_num,train_error,'r',label='Train')
plt.plot(test_num,test_error,'g',label='Test')
plt.xlabel('m')
plt.ylabel('error')
plt.legend()
plt.show()


        
    