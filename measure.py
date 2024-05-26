#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:42:47 2024

@author: anirban
"""

import numpy as np
def measure_new(img ,lut1, lut2, guessFlux = 100, guessmux = 0, guessmuy=0 , guessAlphax =3, guessAlphay =3, guessAlphaxy =0 , counter_target = 100, distort = 0, fixBkg=0, back_calc=0):
    #Shape the image properly
    img = np.array(img)
    sizey, sizex = np.shape(img)
    
    #Resize the cutout if too big
    if(sizex > 100):
        midX = int(round(sizex/2.0))
        img = img[: , midX-50: midX+50]
    if(sizey > 100):
        midY = int(round(sizey/2.0))
        img = img[midY-50: midY+50, :]
    if(sizex< 20 or sizey<20):
        return None, None, None,None, None, None, None, None, None, None
    
    img[img<=0]= np.median(img)
    #If guess sigmas it too small or large
    if(guessAlphax< 0.9):
        guessAlphax = 0.9
    if(guessAlphax> 10):
        guessAlphax = 10
        
    if(guessAlphay< 0.9):
        guessAlphay = 0.9
    if(guessAlphay>10):
        guessAlphay = 10
        
    if(abs(guessAlphaxy)> 100):
        guessAlphaxy = 0
    
    
    #Now define meshgrid for calculation
    sizey, sizex = np.shape(img)
    x = np.linspace(0, sizex-1, sizex)
    y = np.linspace(0, sizey-1, sizey)
    x= x -sizex/2.0 + 0.5 
    y= y -sizey/2.0 + 0.5 
    x, y = np.meshgrid(x, y)
    
    #Initialize variables
    delSigxx = 9999
    delSigyy = 9999

    prevSigxx = 9999
    prevSigyy = 9999
    
    alphax = guessAlphax
    alphay = guessAlphay
    alphaxy = guessAlphaxy
    sigxx_calc = sigyy_calc = sigxy_calc = 0
    mux_calc = guessmux
    muy_calc= guessmuy
    flux_calc = 0 
    e1 = e2 = 0.0
    
    med = np.median(img)
    if(fixBkg==0):
        back_calc = np.median(img)
    
    total = np.sum(img)
    counter = 0
    
    
    
    #Loop until convergence
    
    while( (abs(delSigxx)>0.001 or abs(delSigyy)> 0.001) and counter<counter_target):
        #print (alphax, alphay,alphaxy, counter)
        #Correct alphxy to avoid nans
        while( (alphaxy/(alphax*alphay))**2  >= 1):
            #print (alphax, alphay, alphaxy)
            if(alphaxy > 0 ):
                alphaxy = alphaxy - 0.1
            if (alphaxy < 0):
                alphaxy = alphaxy + 0.1
            if (abs(alphaxy) > 5000):
                counter = counter_target+999
                break
        #If size too large exit
        if(abs(alphaxy)> sizex/2  or abs(alphax) > sizex/3 or abs(alphay)> sizey/3):
            counter = counter_target+999
            break
        
        arbConst = 2*(1- (alphaxy/(alphax*alphay))**2 )  
        A =1/(2*np.pi*alphax*alphay*np.sqrt(1- (alphaxy/(alphax*alphay))**2 ))
        k=(A * np.exp(-((x-mux_calc)**2/(arbConst*alphax**2)+ (y-muy_calc)**2/(arbConst*alphay**2) - 2*alphaxy*(y-muy_calc)*(x-mux_calc)/(arbConst* alphax**2 * alphay**2 ) )))
        
        #Apply distortion correction
        if(distort == 1):
            sqrtImg = np.sqrt(img)
            q= np.abs(img-back_calc)/sqrtImg
            q2 = q*q
            q3 = q2*q
            temp1 = img - back_calc + 1.41421* sqrtImg*(0.477/ np.exp(0.928*q +0.247*q2 + 0.04*q3))
            temp2 = 1.41421* sqrtImg*(0.477/ np.exp(0.551*q - 0.06*q2 + 0.003*q3))
            img1 = img - back_calc
            img1[img1>=0] = temp1[img1>=0]
            img1[img1<0] = temp2[img1<0]
        else:
            img1 = img - back_calc
        
        t1= np.sum(x*y* (img1) * k)
        t2 = np.sum(k*(img1))
        t3 = np.sum(x*(img1)*k)
        t4 = np.sum(y*(img1)*k)
        t5 = np.sum(x * x * (img1) * k)
        t6 = np.sum(y * y * (img1) * k)
        t7 = np.sum(k**2)
        
        mux_calc = t3/t2
        muy_calc = t4/t2
        flux_calc = t2/t7
        
        
        sigxy_calc = (t1/t2) - (t3*t4)/(t2*t2)
        sigxx_calc = (t5/t2) - (t3*t3) / (t2*t2)
        sigyy_calc = (t6/t2) - (t4*t4) / (t2*t2)
        
        
        #If negative sigmas, break
        if(sigxx_calc<0 or sigyy_calc<0):
            counter = counter_target+999
            break
        
        e1 = (sigxx_calc - sigyy_calc)/(sigxx_calc + sigyy_calc)
        e2 = 2*sigxy_calc/(sigxx_calc + sigyy_calc)
        #ellip = np.sqrt(e1*e1 + e2*e2)
        
        if(med != 0 and fixBkg==0):
            back_calc = (total - flux_calc)/ (sizex*sizey)
            
        delSigxx = prevSigxx - sigxx_calc
        delSigyy = prevSigyy - sigyy_calc
        
        prevSigxx = sigxx_calc
        prevSigyy = sigyy_calc
        
        alphax = sigxx_calc*2
        alphay = sigyy_calc*2
        if(alphax <0.9 ):
            alphax = 0.9
        if(alphay <0.9 ):
            alphay = 0.9
        alphax = np.sqrt(alphax)
        alphay = np.sqrt(alphay)
        alphaxy = 2.0*sigxy_calc
       
        
        counter += 1
        
    
    #If failed convergence, return None         
    if(counter == (counter_target+999)):
        return None, None, None,None, None, None, None, None, None, None
    else:
        if((guessAlphax**2 *guessAlphay**2 - guessAlphaxy**2)<=0 or guessFlux<=0):
            return None, None, None,None, None, None, None, None, None, None
        
        if((med< 0 or back_calc<0) and distort == 1):
            return None, None, None,None, None, None, None, None, None, None
        #Appy correction if distorted
        sqrtba = np.sqrt(med)*3.14*np.sqrt(guessAlphax**2 *guessAlphay**2 - guessAlphaxy**2)
        fbysqrtba = np.log10(guessFlux/sqrtba)
        if(distort == 1 and counter_target == 1):
            
            lut = lut1
            if(fbysqrtba< -3.275):
                index = 0
                sigma_factor = lut[index,2]
                flux_bias= lut[index,1]  * sqrtba
            elif(fbysqrtba> 2.1):
                index = 217
                sigma_factor = lut[index,2]
                flux_bias= lut[index,1]  * sqrtba
            else:
                index = int((fbysqrtba + 3.3)/0.025)
                sigma_factor = np.interp(fbysqrtba,lut[index-1:index+2,0], lut[index-1:index+2,2] )
                flux_bias= np.interp(fbysqrtba,lut[index-1:index+2,0], lut[index-1:index+2,1] ) * sqrtba
            flux_calc -= flux_bias
            sigxx_calc -= sigma_factor*(guessAlphax)**2
            sigyy_calc -= sigma_factor*(guessAlphay)**2
            sigxy_calc -= sigma_factor*(guessAlphaxy)    
                
        elif(distort == 1 and counter_target > 1):
            lut = lut2
            if(fbysqrtba< 1.225):
                index = 0
                sigma_factor = lut[index,2]
                flux_bias= lut[index,1]  * sqrtba
            elif(fbysqrtba> 2.825):
                index = 34
                sigma_factor = lut[index,2]
                flux_bias= lut[index,1]  * sqrtba
            else:
                index = int((fbysqrtba -1.175)/0.05)
                sigma_factor = np.interp(fbysqrtba,lut[index-1:index+2,0], lut[index-1:index+2,2] )
                flux_bias= np.interp(fbysqrtba,lut[index-1:index+2,0], lut[index-1:index+2,1] ) * sqrtba
            flux_calc -= flux_bias
            sigxx_calc -= sigma_factor*(guessAlphax)**2
            sigyy_calc -= sigma_factor*(guessAlphay)**2
            sigxy_calc -= sigma_factor*(guessAlphaxy)
            
        
        return flux_calc, mux_calc, muy_calc, e1, e2, back_calc, np.sqrt(sigxx_calc + sigyy_calc), sigxx_calc, sigyy_calc, sigxy_calc




#Guess flux, centroid and size are used only when using forced measurement 


from astropy.io import fits 
lut1 = np.load('/home/anirban/calib_final.npy')  #CHANGE PATH HERE AS REQUIRED
lut2 = np.load('/home/anirban/calib_final_inf.npy') #CHANGE PATH HERE AS REQUIRED
f=fits.open('/home/anirban/example.fits')     #CHANGE PATH HERE AS REQUIRED
cut = f[0].data
f.close()

#############################################
#USAGE SYNTAX
#  measure_new(cut(the image array) , lut1 (look up table 1 as defined above), 
#                              lut2(as defined above), guess counts (optional), 
#                             guess centroid x (optional), guess centroid y (optional),
#                             guess alpha x (optional), guess alpha y (optional),
#                             guess alpha xy (optional), number of iteration (optional), distortion (1=yes 0=no, optional),
#                             fix bkg (1=yes, 0= no, optional), level of bkg to be fixed at (optional))
#
# Guess alpha x = sqrt(2* sigma_xx returned from measure_new)  
# Guess alpha y = sqrt(2* sigma_yy returned from measure_new) 
# Guess alpha xy = 2* sigma_xy output of measure_new
###############################################




# =============================================================================
# #Example 1 (measuring until convergence)
# counts, mux, muy, e1, e2, bkg , size , sigxx, sigyy, sigxy = measure_new(cut, lut1, lut2)
# print ("Counts ", counts)  #884203.4038681877
# print ("Size in pixels", size) #7.698558724065786
# print ("Background Level cts/pix", bkg) #107.31165961318123
# print ("e1", e1) #0.1723999781775083
# print ("e2", e2) #-0.10412013679646344
# print ("mux", mux) #-0.03094049888449857
# print ("muy", muy)#-0.09803823049271888
# print ("Sigma xx (weighted)", sigxx) #34.74278748134318
# print ("Sigma yy (weighted)", sigyy)#24.525018946546254
# print ("Sigma xy (weighted)", sigxy)#-3.0854860564490814
# 
# =============================================================================



# =============================================================================
# #Example 2 (forced measurement without distortion)
# counts, mux, muy, e1, e2, bkg , size , sigxx, sigyy, sigxy = measure_new(cut, lut1, lut2, 884244.85, 0, 0, 8, 7.5, -6.17, 1, 0)
# #Guess alpahx = sqrt(2*34.742 ) where 34.742 is obtained from Example 1. We give appx value of 8
# #Guess alpahy = sqrt(2*24.525 ) where 24.525 is obtained from Example 1. We give appx value of 7.5
# #Guess alpahxy = 2*-3.08548 where -3.08548 is obtained from the Example 1. We pass exact value -6.17
# #Guess counts and centroid is also obtained from above case 
# print ("Counts ", counts)#893802.4812067741
# print ("Size in pixels", size)#7.7137704038602815
# print ("Background Level cts/pix", bkg)#106.35175187932258
# print ("e1", e1)#0.11819563966471783
# print ("e2", e2)#-0.1031230773381151
# print ("mux", mux)#-0.023117961219654825
# print ("muy", muy)#-0.06645623072282557
# print ("Sigma xx (weighted)", sigxx)#33.26758039899613
# print ("Sigma yy (weighted)", sigyy)#26.234673444474684
# print ("Sigma xy (weighted)", sigxy)#-3.0680277624461985
# 
# =============================================================================




# =============================================================================
# #Example 3 (forced measurement with distortion)
# counts, mux, muy, e1, e2, bkg , size , sigxx, sigyy, sigxy = measure_new(cut, lut1, lut2, 884244.85, 0, 0, 8, 7.5, -6.17, 1, 1)
# #Guess alpahx = sqrt(2*34.742 ) where 34.742 is obtained from Example 1. We give appx value of 8
# #Guess alpahy = sqrt(2*24.525 ) where 24.525 is obtained from Example 1. We give appx value of 7.5
# #Guess alpahxy = 2*-3.08548 where -3.08548 is obtained from the Example 1. We pass exact value -6.17
# #Guess counts and centroid is also obtained from above case 
# print ("Counts ", counts)#895375.046198461
# print ("Size in pixels", size)#7.706383044402326
# print ("Background Level cts/pix", bkg)#106.34717590448456
# print ("e1", e1)#0.11819476635103246
# print ("e2", e2)#-0.10313805908456995
# print ("mux", mux)#-0.023783234192061596
# print ("muy", muy)#-0.0662605651155278
# print ("Sigma xx (weighted)", sigxx)#33.20775805464459
# print ("Sigma yy (weighted)", sigyy)#26.180581572407075
# print ("Sigma xy (weighted)", sigxy)#-3.0626365966187317
# =============================================================================

