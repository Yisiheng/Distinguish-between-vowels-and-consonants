# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import soundfile as sf
import gudhi
from gudhi.point_cloud import timedelay
import numpy as np




Str='D:\\音标音频\\[f]_1.WAV'

def Read_Audio(Str):
    sig, samplerate = sf.read(Str)
    
    f1 = plt.figure(1)
    plt.plot(sig)   
    return sig, samplerate

def Persistence_Homology(Data,Type_Complex='VR', Max_edge_length=1, Max_dimension=3):
    if Type_Complex=='VR':
        rips_complex = gudhi.RipsComplex(points=Data, max_edge_length=Max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=Max_dimension)
        diag = simplex_tree.persistence(min_persistence=0.01)
        gudhi.plot_persistence_barcode(diag,legend=True)
        gudhi.plot_persistence_diagram(diag,legend=True)
        plt.show()
    elif Type_Complex=='Alpha':
        1
    else:
        1
        
        
        
def MainFunction(Delay=1,Skip=1,Dim=1,Max_edge_length=1,Max_dimension=1):
    Str='D:\单元音\前元音\[æ]_1.WAV'
    sig, samplerate=Read_Audio(Str)
    point_Cloud=timedelay.TimeDelayEmbedding(dim=Dim, delay=Delay, skip=Skip)
    Points=point_Cloud(sig[3500:5000])
    Persistence_Homology(Points,'VR', Max_edge_length, Max_dimension)


def FFTransform(sig,samplerate):
    
    x=np.array(range(len(sig)))
    y=sig
    
    yy=fft(y)   #快速傅里叶变换
    yreal = yy.real    # 获取实数部分
    yimag = yy.imag    # 获取虚数部分

    yf=abs(fft(y))    # 取绝对值
    yf1=abs(fft(y))/len(x)    #归一化处理
    yf2 = yf1[range(int(len(x)/2))]    #由于对称性，只取一半区间

    xf = np.arange(len(y))  # 频率
    xf1 = xf
    xf2 = xf[range(int(len(x)/2))] #取一半区间
    
    fig=plt.figure()
    plt.subplot(211)
    plt.plot(x/samplerate,y) 
    plt.title('Original wave')
     
    plt.subplot(212)
    plt.plot(xf2*(samplerate/len(sig)),yf2,'b')
    plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')
     
    plt.show()
    
    return xf2*(samplerate/len(sig)), yf2 #返回值 频率，振幅
    
def Find_parameter(sig,samplerate): #未完
    x,y=FFTransform(sig,samplerate)
    N=int(len(x)/100)
    
    X=[0]*100
    Y=[0]*100
    
    for i in range(100):
        Y[i]=sum(y[N*i:N*(i+1)])
        X[i]=sum(x[N*i:N*(i+1)])/N
    
    



#test

def Sphere(N=30):
    
    sphere=[]
    for i in range(N):
        for j in range(N):
            theta=i/N*np.pi
            phi=j/N*2*np.pi
            
            x=np.sin(theta)*np.cos(phi)
            y=np.sin(theta)*np.sin(phi)
            z=np.cos(theta)
            
            sphere.append([x,y,z])
            
    sphere=np.array(sphere)
    
    ax = plt.subplot(projection = '3d')
    ax.set_title('3d_image_show')
    ax.scatter(sphere[:,0], sphere[:,1], sphere[:,2], c = 'r')
    
    return sphere


def Sin(N=100):
    
    x=np.linspace(0,2*2*np.pi,2*N)
    y=np.sin(x)
    
    fig=plt.figure()
    plt.scatter(x,y,s=5)
    
    return y

def f(data):
    N=len(data)
    #data=list(data)
    point_Cloud=timedelay.TimeDelayEmbedding(dim=2, delay=int(N/8), skip=1)
    Points=point_Cloud(data)
    
    fig=plt.figure()
    plt.scatter(Points[:,0],Points[:,1])
    
    Persistence_Homology(Points,'VR', 1, 2)
    


















