
# coding: utf-8

# In[25]:

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from dtw import dtw
import math
from sklearn import linear_model
from sklearn.svm import SVR
from statsmodels.tsa.arima_model import ARIMA
df = pd.read_csv('dataset.csv', header = 0)
df=df[(df.component == 'IMIPENEM') | (df.component == 'MEROPENEM') | (df.component == 'GEMIFLOXACIN') | (df.component == 'CEFTAZIDIME') | (df.component == 'LOMEFLOXACIN') | (df.component == 'CEFOXITIN') |(df.component == 'CEFEPIME') | (df.component == 'CEFOTETAN') | (df.component == 'DORIPENEM') | (df.component == 'CEFUROXIME') | (df.component == 'ERTAPENEM') | (df.component == 'CEPHALOTHIN') | (df.component == 'GATIFLOXACIN') | (df.component == 'TROVAFLOXACIN') | (df.component == 'NORFLOXACIN') | (df.component == 'MOXIFLOXACIN') | (df.component == 'CLINAFLOXACIN') | (df.component == "LEVOFLOXACIN") | (df.component == 'CIPROFLOXACIN') | (df.component == 'CEFOTAXIME') | (df.component == "CEFAZOLIN") | (df.component == 'CEFTRIAXONE') | (df.component == 'CEFTAZIDIME')] 
df=df[(df.organism == 'E. coli') | (df.organism == 'Klebsiella oxytoca') | (df.organism == 'Klebsiella pneumoniae')]
#df = df[(df.organism != 'MRSA') & (df.organism != 'MSSA') & (df.organism != "Staphylococcus aureus")]
df=df.set_index(np.arange(0,df.shape[0]))
print(df.shape)
df.head()


# In[26]:

set(df.organism)


# In[27]:

df=df.set_index(np.arange(0,df.shape[0]))
ABpairs = []
for i in range(0, df.shape[0]):
    pair = str(df.component[i]) + str(df.organism[i]) + str(df.hospitalid[i])
    if pair not in ABpairs:
        ABpairs.append(pair)
    
print(len(ABpairs))
##print(len(set(df.component)))
##print(len(set(df.organism)))


# In[28]:

#sigPairs = pd.read_csv('ABratings.csv')
#experiment = pd.merge(sigPairs, df, on=['component', 'organism'])
#experiment = experiment[(experiment.rating == 2)]
#df = experiment
#df.head()


# In[29]:

#parameters
samples = 20
targetYears = [2014, 2015, 2016]
yearsAhead = 1
totalYears = 12 ##### experiment with
nYears = 6
reportsY = 1
reportsy9 = 1
reportsy8 = 1
priorReports = 0  ##### experiment with
#calculations
df = df[df["Total Tests (by organism)"] >=samples]
years9 = []
years1 = []
for Y in targetYears:
    years9.append(Y-yearsAhead)
    years1.append(Y-(yearsAhead-1)-totalYears)
antibiotic = []
bacteria = []
location = []
tY = []
y9 = []
y1 = []

r = []
rY = []
ry9 = []
ry8 = []

actualY = []
actual9 = []
actual8 = []

sdY = []
sd9 = []
sd8 = []

RegLinY = []
RegPolyY = []
SVRlinY = []
SVRrbfY = []
SVRsigY = []

RegLinYw = []
RegPolyYw = []
SVRlinYw = []
SVRrbfYw = []
SVRsigYw = []

RegLin9 = []
RegPoly9 = []
SVRlin9 = []
SVRrbf9 = []
SVRsig9 = []

RegLin9w = []
RegPoly9w = []
SVRlin9w = []
SVRrbf9w = []
SVRsig9w = []

LR9e = []
PR9e = []
LS9e = []
GS9e = []
SS9e = []
pm9e = []
LR9ew = []
PR9ew = []
LS9ew = []
GS9ew = []
SS9ew = []

LR9score = []
PR9score = []
LS9score = []
GS9score = []
SS9score = []
LR9scorew = []
PR9scorew = []
LS9scorew = []
GS9scorew = []
SS9scorew = []

predIntLR = []
predIntLS = []
predIntGS = []
predIntSS = []
predIntPR = []
predIntLRw = []
predIntLSw = []
predIntGSw = []
predIntSSw = []
predIntPRw = []

modelFitLR = []
modelFitLS = []
modelFitGS = []
modelFitSS = []
modelFitPR = []
modelFitLRw = []
modelFitLSw = []
modelFitGSw = []
modelFitSSw = []
modelFitPRw = []

suscSet = []
yearSet = []

arima1 = []
arima2 = []
arima3 = []
arima4 = []
arima5 = []
arima6 = []

arima1s = []
arima2s = []
arima3s = []
arima4s = []
arima5s = []
arima6s = []

for org in set(df.organism):
    for ab in set(df.component):
        tdf = df[(df.component == ab) & (df.organism == org)]
        for Y in range(0,len(targetYears)):
            ttdf = tdf[(tdf['Report Year'] != years9[Y]) & (tdf['Report Year'] != targetYears[Y]) & (tdf['Report Year'] >= years1[Y])]
            ttdfY = tdf[(tdf['Report Year'] == targetYears[Y])]
            ttdf9 = tdf[(tdf['Report Year'] == years9[Y])]
            ttdf8 = tdf[(tdf['Report Year'] == (years9[Y]-1))]
            if ((ttdf.shape[0]>=priorReports) & (len(set(ttdf["Report Year"])) >= nYears) & (ttdfY.shape[0]>=reportsY) & (ttdf9.shape[0]>=reportsy9) & (ttdf8.shape[0]>=reportsy8)):
                ttdf=ttdf.set_index(np.arange(0,ttdf.shape[0]))
                ttdfY=ttdfY.set_index(np.arange(0,ttdfY.shape[0]))
                ttdf9=ttdf9.set_index(np.arange(0,ttdf9.shape[0]))
                ttdf8=ttdf8.set_index(np.arange(0,ttdf8.shape[0])) 

                antibiotic.append(ab)
                bacteria.append(org)
                r.append(ttdf.shape[0])
                rY.append(ttdfY.shape[0])
                ry9.append(ttdf9.shape[0])
                ry8.append(ttdf8.shape[0])
                tY.append(targetYears[Y])
                y9.append(years9[Y])
                y1.append(years1[Y])

                tempdf9=tdf[(tdf['Report Year'] <= years9[Y]) & (tdf['Report Year'] >= (years1[Y]))]
                tempdf8=tdf[(tdf['Report Year'] <= (years9[Y]-1)) & (tdf['Report Year'] >= (years1[Y]-1))]
                tempdf9=tempdf9.set_index(np.arange(0,tempdf9.shape[0]))
                tempdf8=tempdf8.set_index(np.arange(0,tempdf8.shape[0]))

                #Calculate actual and sd
                year9 = []
                susc9 = []
                weight9 = []
                for i in range(0, len(tempdf9)):
                    year9.append(tempdf9["Report Year"][i])
                    susc9.append(tempdf9["Indicator Value (Pct)"][i])
                    weight9.append(tempdf9["Total Tests (by organism)"][i])
                year8 = []
                susc8 = []
                weight8 = []
                for i in range(0, len(tempdf8)):
                    year8.append(tempdf8["Report Year"][i])
                    susc8.append(tempdf8["Indicator Value (Pct)"][i])
                    weight8.append(tempdf8["Total Tests (by organism)"][i])

                year9T = np.transpose(np.matrix(year9))
                susc9T = np.transpose(np.matrix(susc9))
                year8T = np.transpose(np.matrix(year8))
                susc8T = np.transpose(np.matrix(susc8))
                weight8T = np.transpose(np.matrix(weight8))
                weight9T = np.transpose(np.matrix(weight9))
                aY = 0
                a9 = 0
                a8 = 0
                sY = 0
                s9 = 0
                s8 = 0
                for a in range(0, ttdfY.shape[0]):
                    aY = aY + ttdfY["Total Tests (by organism)"][a] * ttdfY["Indicator Value (Pct)"][a]
                    #sY = sY + (a-(aY/sum(ttdfY["Total Tests (by organism)"])))**(2)
                    sY = sY + (ttdfY["Indicator Value (Pct)"][a] - sum(ttdfY["Indicator Value (Pct)"])/ttdfY.shape[0])**2
                sdY.append((sY/ttdfY.shape[0])**(1/2))  
                actualY.append(aY/sum(ttdfY["Total Tests (by organism)"]))
                for a in range(0, ttdf9.shape[0]):
                    a9 = a9 + ttdf9["Total Tests (by organism)"][a] * ttdf9["Indicator Value (Pct)"][a]
                    s9 = s9 + (ttdf9["Indicator Value (Pct)"][a] - sum(ttdf9["Indicator Value (Pct)"])/ttdf9.shape[0])**2
                    #s9 = s9 + (a-(a9/sum(ttdf9["Total Tests (by organism)"])))**(2)
                sd9.append((s9/ttdf9.shape[0])**(1/2))
                actual9.append(a9/sum(ttdf9["Total Tests (by organism)"]))
                for a in range(0, ttdf8.shape[0]):
                    a8 = a8 + ttdf8["Total Tests (by organism)"][a] * ttdf8["Indicator Value (Pct)"][a]
                    s8 = s8 + (ttdf8["Indicator Value (Pct)"][a] - sum(ttdf8["Indicator Value (Pct)"])/ttdf8.shape[0])**2
                    #s8 = s8 + (a-(a8/sum(ttdf8["Total Tests (by organism)"])))**(2)      
                sd8.append((s8/ttdf8.shape[0])**(1/2))
                actual8.append(a8/sum(ttdf8["Total Tests (by organism)"]))

                #Previous Fit
                regr8 = linear_model.LinearRegression()
                regr8.fit(year8T, susc8T)
                RegLin9.append(regr8.predict(years9[Y])[0][0])
                regr8w = linear_model.LinearRegression()
                regr8w.fit(year8T, susc8T, sample_weight = weight8)
                RegLin9w.append(regr8w.predict(years9[Y])[0][0])

                clfL8 = SVR(kernel = "linear")
                clfL8.fit(year8T.reshape(-1,1), susc8)
                SVRlin9.append(clfL8.predict(years9[Y])[0])
                clfL8w = SVR(kernel = "linear")
                clfL8w.fit(year8T.reshape(-1,1), susc8, sample_weight = weight8)
                SVRlin9w.append(clfL8w.predict(years9[Y])[0])

                clf8 = SVR(kernel = "rbf")
                clf8.fit(year8T.reshape(-1,1), susc8)
                SVRrbf9.append(clf8.predict(years9[Y])[0])
                clf8w = SVR(kernel = "rbf")
                clf8w.fit(year8T.reshape(-1,1), susc8, sample_weight = weight8)
                SVRrbf9w.append(clf8w.predict(years9[Y])[0])

                clfS8 = SVR(kernel = "sigmoid")
                clfS8.fit(year8T.reshape(-1,1), susc8)
                SVRsig9.append(clfS8.predict(years9[Y])[0])
                clfS8w = SVR(kernel = "sigmoid")
                clfS8w.fit(year8T.reshape(-1,1), susc8, sample_weight = weight8)
                SVRsig9w.append(clfS8w.predict(years9[Y])[0])

                z8 = np.polyfit(year8, susc8, 2)
                p8 = np.poly1d(z8)
                RegPoly9.append(p8(years9[Y]))
                z8w = np.polyfit(year8, susc8, 2, w = weight8)
                p8w = np.poly1d(z8w)
                RegPoly9w.append(p8w(years9[Y]))

                #Current Fit
                regr9 = linear_model.LinearRegression()
                regr9.fit(year9T, susc9T)
                RegLinY.append(regr9.predict(targetYears[Y])[0][0])
                regr9w = linear_model.LinearRegression()
                regr9w.fit(year9T, susc9T, sample_weight = weight9)
                RegLinYw.append(regr9w.predict(targetYears[Y])[0][0])

                clfL9 = SVR(kernel = "linear")
                clfL9.fit(year9T.reshape(-1,1), susc9)
                SVRlinY.append(clfL9.predict(targetYears[Y])[0])
                clfL9w = SVR(kernel = "linear")
                clfL9w.fit(year9T.reshape(-1,1), susc9, sample_weight = weight9)
                SVRlinYw.append(clfL9w.predict(targetYears[Y])[0])

                clf9 = SVR(kernel = "rbf")
                clf9.fit(year9T.reshape(-1,1), susc9)
                SVRrbfY.append(clf9.predict(targetYears[Y])[0])
                clf9w = SVR(kernel = "rbf")
                clf9w.fit(year9T.reshape(-1,1), susc9, sample_weight = weight9)
                SVRrbfYw.append(clf9w.predict(targetYears[Y])[0])

                clfS9 = SVR(kernel = "sigmoid")
                clfS9.fit(year9T.reshape(-1,1), susc9)
                SVRsigY.append(clfS9.predict(targetYears[Y])[0])
                clfS9w = SVR(kernel = "sigmoid")
                clfS9w.fit(year9T.reshape(-1,1), susc9, sample_weight = weight9)
                SVRsigYw.append(clfS9w.predict(targetYears[Y])[0])

                z9 = np.polyfit(year9, susc9, 2)
                p9 = np.poly1d(z9)
                RegPolyY.append(p9(targetYears[Y]))
                z9w = np.polyfit(year9, susc9, 2, w = weight9)
                p9w = np.poly1d(z9w)
                RegPolyYw.append(p9w(targetYears[Y]))
                
print("done")


# In[30]:

# Calculate Differences                

LR9d = []
PR9d = []
LS9d = []
GS9d = []
SS9d = []
LR9wd = []
PR9wd = []
LS9wd = []
GS9wd = []
SS9wd = []
pm9d = []
arima1sd = []
arima2sd = []
arima3sd = []
arima4sd = []
arima6sd = []
#for c in range(0, len(actual9)):
#    LR9d.append(abs(RegLin9[c]-actual9[c]))  
#    PR9d.append(abs(RegPoly9[c]-actual9[c]))
#    LS9d.append(abs(SVRlin9[c]-actual9[c]))
#    GS9d.append(abs(SVRrbf9[c]-actual9[c]))
#    SS9d.append(abs(SVRsig9[c]-actual9[c]))
#    LR9wd.append(abs(RegLin9w[c]-actual9[c]))  
#    PR9wd.append(abs(RegPoly9w[c]-actual9[c]))
#    LS9wd.append(abs(SVRlin9w[c]-actual9[c]))
#    GS9wd.append(abs(SVRrbf9w[c]-actual9[c]))
#    SS9wd.append(abs(SVRsig9w[c]-actual9[c]))
#    pm9d.append(abs(actual9[c]-actual8[c]))
#    arima1sd.append(abs(arima1s[c]-actual9[c]))
#    arima2sd.append(abs(arima2s[c]-actual9[c]))
#    arima3sd.append(abs(arima3s[c]-actual9[c]))
#    arima4sd.append(abs(arima4s[c]-actual9[c]))
#    arima6sd.append(abs(arima6s[c]-actual9[c]))
for c in range(0, len(actual9)):
    LR9d.append((RegLin9[c]-actual9[c])**2)  
    PR9d.append((RegPoly9[c]-actual9[c])**2)
    LS9d.append((SVRlin9[c]-actual9[c])**2)
    GS9d.append((SVRrbf9[c]-actual9[c])**2)
    SS9d.append((SVRsig9[c]-actual9[c])**2)
    LR9wd.append((RegLin9w[c]-actual9[c])**2)  
    PR9wd.append((RegPoly9w[c]-actual9[c])**2)
    LS9wd.append((SVRlin9w[c]-actual9[c])**2)
    GS9wd.append((SVRrbf9w[c]-actual9[c])**2)
    SS9wd.append((SVRsig9w[c]-actual9[c])**2)
    pm9d.append((actual9[c]-actual8[c])**2)

#PYPER
upperbound = []
upperboundw = []
upperboundAll = []
upperboundAllw = []
upperboundP = []
upperboundPw = []
upperboundArima =[]
upperboundMix = []
upperboundL = []
upperboundLw = []
PYPER = []
PYPERmethod = []
PYPERw = []
PYPERmethodw = []
PYPERr = []
PYPERrw = []
PYPERallw = []
PYPERall = []
PYPERp = []
PYPERpw = []
for i in range(0, len(actualY)):
    upperbound.append(min(LR9d[i], LS9d[i], GS9d[i]))
    upperboundL.append(min(LR9d[i], LS9d[i]))
    upperboundLw.append(min(LR9wd[i], LS9wd[i]))
    upperboundw.append(min(LR9wd[i], LS9wd[i], GS9wd[i]))
    upperboundAll.append(min(LR9d[i], LS9d[i], GS9d[i], PR9d[i], SS9d[i]))
    upperboundAllw.append(min(LR9wd[i], LS9wd[i], GS9wd[i], PR9wd[i], SS9wd[i]))
    upperboundP.append(min(LR9d[i], LS9d[i], GS9d[i], pm9d[i]))
    upperboundPw.append(min(LR9wd[i], LS9wd[i], GS9wd[i], pm9d[i]))
    
    if (upperboundLw[i] == LR9wd[i]):
        PYPERrw.append(RegLinYw[i])
    elif (upperboundLw[i] == LS9wd[i]):
        PYPERrw.append(SVRlinYw[i])

    if (upperboundL[i] == LR9d[i]):
        PYPERr.append(RegLinY[i])
    elif (upperboundL[i] == LS9d[i]):
        PYPERr.append(SVRlinY[i])
        
    if (upperboundw[i] == LR9wd[i]):
        PYPERw.append(RegLinYw[i])
    elif (upperboundw[i] == LS9wd[i]):
        PYPERw.append(SVRlinYw[i])
    elif (upperboundw[i] == GS9wd[i]):
        PYPERw.append(SVRrbfYw[i])

    if (upperbound[i] == LR9d[i]):
        PYPER.append(RegLinY[i])
    elif (upperbound[i] == LS9d[i]):
        PYPER.append(SVRlinY[i])
    elif (upperbound[i] == GS9d[i]):
        PYPER.append(SVRrbfY[i])
        
    if (upperboundAllw[i] == LR9wd[i]):
        PYPERallw.append(RegLinYw[i])
    elif (upperboundAllw[i] == LS9wd[i]):
        PYPERallw.append(SVRlinYw[i])
    elif (upperboundAllw[i] == GS9wd[i]):
        PYPERallw.append(SVRrbfYw[i])
    elif (upperboundAllw[i] == PR9wd[i]):
        PYPERallw.append(RegPolyYw[i])
    elif(upperboundAllw[i] == SS9wd[i]):
        PYPERallw.append(SVRsigYw[i])

    if (upperboundAll[i] == LR9d[i]):
        PYPERall.append(RegLinY[i])
    elif (upperboundAll[i] == LS9d[i]):
        PYPERall.append(SVRlinY[i])
    elif (upperboundAll[i] == GS9d[i]):
        PYPERall.append(SVRrbfY[i])
    elif (upperboundAll[i] == PR9d[i]):
        PYPERall.append(RegPolyY[i])
    elif(upperboundAll[i] == SS9d[i]):
        PYPERall.append(SVRsigY[i])
     
    if (upperboundPw[i] == pm9d[i]):
        PYPERpw.append(actual9[i])
    elif (upperboundPw[i] == LR9wd[i]):
        PYPERpw.append(RegLinYw[i])
    elif (upperboundPw[i] == LS9wd[i]):
        PYPERpw.append(SVRlinYw[i])
    elif (upperboundPw[i] == GS9wd[i]):
        PYPERpw.append(SVRrbfYw[i])
    elif (upperboundPw[i] == PR9wd[i]):
        PYPERpw.append(RegPolyYw[i])
    elif(upperboundPw[i] == SS9wd[i]):
        PYPERpw.append(SVRsigYw[i])

    if (upperboundP[i] == pm9d[i]):
        PYPERp.append(actual9[i])
    elif (upperboundP[i] == LR9d[i]):
        PYPERp.append(RegLinY[i])
    elif (upperboundP[i] == LS9d[i]):
        PYPERp.append(SVRlinY[i])
    elif (upperboundP[i] == GS9d[i]):
        PYPERp.append(SVRrbfY[i])
    elif (upperboundP[i] == PR9d[i]):
        PYPERp.append(RegPolyY[i])
    elif(upperboundP[i] == SS9d[i]):
        PYPERp.append(SVRsigY[i])
        
        
PYPERed = []
PYPERse = []
PYPERsd = []
PYPER1 = []
PYPERedw = []
PYPERsew = []
PYPERsdw = []
PYPER1w = []
for L in range(0, len(PYPER)):
    if pm9d[L]<=10/(ry9[L]**(1/2)):
        PYPERed.append(actual9[L])
    else:
        PYPERed.append(PYPER[L])
    if pm9d[L]<=sd9[L]/(ry9[L]**(1/2)):
        PYPERse.append(actual9[L])
    else:
        PYPERse.append(PYPER[L])
    if pm9d[L]<=sd9[L]:
        PYPERsd.append(actual9[L])
    else:
        PYPERsd.append(PYPER[L])
    if pm9d[L]<= 1:
        PYPER1.append(actual9[L])
    else:
        PYPER1.append(PYPER[L])
    if pm9d[L]<=10/(ry9[L]**(1/2)):
        PYPERedw.append(actual9[L])
    else:
        PYPERedw.append(PYPERw[L])
    if pm9d[L]<=sd9[L]/(ry9[L]**(1/2)):
        PYPERsew.append(actual9[L])
    else:
        PYPERsew.append(PYPERw[L])
    if pm9d[L]<=sd9[L]:
        PYPERsdw.append(actual9[L])
    else:
        PYPERsdw.append(PYPERw[L])
    if pm9d[L]<= 1:
        PYPER1w.append(actual9[L])
    else:
        PYPER1w.append(PYPERw[L])
    

#Adjust for out of bound values
PYP = []
for item in PYPER:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    PYP.append(item)
PYPw = []
for item in PYPERw:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    PYPw.append(item)
PYPr = []
for item in PYPERr:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    PYPr.append(item)
PYPrw = []
for item in PYPERrw:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    PYPrw.append(item)
PYPa = []
for item in PYPERall:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    PYPa.append(item)
PYPaw = []
for item in PYPERallw:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    PYPaw.append(item)
PYPp = []
for item in PYPERp:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    PYPp.append(item)
PYPpw = []
for item in PYPERpw:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    PYPpw.append(item)
LRY = []
for item in RegLinY:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    LRY.append(item)
PRY = []
for item in RegPolyY:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    PRY.append(item)
LSY = []
for item in SVRlinY:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    LSY.append(item)
GSY = []
for item in SVRrbfY:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    GSY.append(item)
SSY = []
for item in SVRsigY:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    SSY.append(item)
LRYw = []
for item in RegLinYw:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    LRYw.append(item)
PRYw = []
for item in RegPolyYw:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    PRYw.append(item)
LSYw = []
for item in SVRlinYw:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    LSYw.append(item)
GSYw = []
for item in SVRrbfYw:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    GSYw.append(item)
SSYw = []
for item in SVRsigYw:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    SSYw.append(item)    
Ped = []
for item in PYPERed:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    Ped.append(item)
Pse = []
for item in PYPERse:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    Pse.append(item)
Psd = []
for item in PYPERsd:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    Psd.append(item)
P1 = []
for item in PYPER1:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    P1.append(item)
Pedw = []
for item in PYPERedw:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    Pedw.append(item)
Psew = []
for item in PYPERsew:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    Psew.append(item)
Psdw = []
for item in PYPERsdw:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    Psdw.append(item)
P1w = []
for item in PYPER1w:
    if item < 0:
        item = 0
    if item > 100:
        item = 100
    P1w.append(item)
    
PYPd = []
PYPd2 = []
PYPcount = []
for c in range(0, len(actualY)):
    PYPd.append(abs(PYP[c]-actualY[c]))
    PYPd2.append((PYP[c]-actualY[c])**2)
    if PYPd[c] <= 5:
        PYPcount.append(1)
    else:
        PYPcount.append(0)
PYPAd = []
PYPAd2 = []
PYPAcount = []
for c in range(0, len(actualY)):
    PYPAd.append(abs(PYPa[c]-actualY[c]))
    PYPAd2.append((PYPa[c]-actualY[c])**2)
    if PYPAd[c] <= 5:
        PYPAcount.append(1)
    else:
        PYPAcount.append(0)
PYPrd = []
PYPrd2 = []
PYPrcount = []
for c in range(0, len(actualY)):
    PYPrd.append(abs(PYPr[c]-actualY[c]))
    PYPrd2.append((PYPr[c]-actualY[c])**2)
    if PYPrd[c] <= 5:
        PYPrcount.append(1)
    else:
        PYPrcount.append(0)
PYPpd = []
PYPpd2 = []
PYPpcount = []
for c in range(0, len(actualY)):
    PYPpd.append(abs(PYPp[c]-actualY[c]))
    PYPpd2.append((PYPp[c]-actualY[c])**2)
    if PYPpd[c] <= 5:
        PYPpcount.append(1)
    else:
        PYPpcount.append(0)
LRYd = []
LRYd2 = []
LRYcount = []
for c in range(0, len(actualY)):
    LRYd.append(abs(LRY[c]-actualY[c]))
    LRYd2.append((LRY[c]-actualY[c])**2)
    if LRYd[c] <= 5:
        LRYcount.append(1)
    else:
        LRYcount.append(0)
PRYd = []
PRYd2 = []
PRYcount = []
for c in range(0, len(actualY)):
    PRYd.append(abs(PRY[c]-actualY[c]))
    PRYd2.append((PRY[c]-actualY[c])**2)
    if PRYd[c] <= 5:
        PRYcount.append(1)
    else:
        PRYcount.append(0)
LSYd = []
LSYd2 = []
LSYcount = []
for c in range(0, len(actualY)):
    LSYd.append(abs(LSY[c]-actualY[c]))
    LSYd2.append((LSY[c]-actualY[c])**2)
    if LSYd[c] <= 5:
        LSYcount.append(1)
    else:
        LSYcount.append(0)
GSYd = []
GSYd2 = []
GSYcount = []
for c in range(0, len(actualY)):
    GSYd.append(abs(GSY[c]-actualY[c]))
    GSYd2.append((GSY[c]-actualY[c])**2)
    if GSYd[c] <= 5:
        GSYcount.append(1)
    else:
        GSYcount.append(0)
SSYd = []
SSYd2 = []
SSYcount = []
for c in range(0, len(actualY)):
    SSYd.append(abs(SSY[c]-actualY[c]))
    SSYd2.append((SSY[c]-actualY[c])**2)
    if SSYd[c] <= 5:
        SSYcount.append(1)
    else:
        SSYcount.append(0)
PEDd = []
PEDd2 = []
PEDcount = []
for c in range(0, len(actualY)):
    PEDd.append(abs(Ped[c]-actualY[c]))
    PEDd2.append((Ped[c]-actualY[c])**2)
    if PEDd[c] <= 5:
        PEDcount.append(1)
    else:
        PEDcount.append(0)
PSEd = []
PSEd2 = []
PSEcount = []
for c in range(0, len(actualY)):
    PSEd.append(abs(Pse[c]-actualY[c]))
    PSEd2.append((Pse[c]-actualY[c])**2)
    if PSEd[c] <= 5:
        PSEcount.append(1)
    else:
        PSEcount.append(0)
PSDd = []
PSDd2 = []
PSDcount = []
for c in range(0, len(actualY)):
    PSDd.append(abs(Psd[c]-actualY[c]))
    PSDd2.append((Psd[c]-actualY[c])**2)
    if PSDd[c] <= 5:
        PSDcount.append(1)
    else:
        PSDcount.append(0)
P1d = []
P1d2 = []
P1count = []
for c in range(0, len(actualY)):
    P1d.append(abs(P1[c]-actualY[c]))
    P1d2.append((P1[c]-actualY[c])**2)
    if P1d[c] <= 5:
        P1count.append(1)
    else:
        P1count.append(0)
    
PYPwd = []
PYPwd2 = []
PYPwcount = []
for c in range(0, len(actualY)):
    PYPwd.append(abs(PYPw[c]-actualY[c]))
    PYPwd2.append((PYPw[c]-actualY[c])**2)
    if PYPwd[c] <= 5:
        PYPwcount.append(1)
    else:
        PYPwcount.append(0)
PYPawd = []
PYPawd2 = []
PYPawcount = []
for c in range(0, len(actualY)):
    PYPawd.append(abs(PYPaw[c]-actualY[c]))
    PYPawd2.append((PYPaw[c]-actualY[c])**2)
    if PYPawd[c] <= 5:
        PYPawcount.append(1)
    else:
        PYPawcount.append(0)
PYPrwd = []
PYPrwd2 = []
PYPrwcount = []
for c in range(0, len(actualY)):
    PYPrwd.append(abs(PYPrw[c]-actualY[c]))
    PYPrwd2.append((PYPrw[c]-actualY[c])**2)
    if PYPrwd[c] <= 5:
        PYPrwcount.append(1)
    else:
        PYPrwcount.append(0)
PYPpwd = []
PYPpwd2 = []
PYPpwcount = []
for c in range(0, len(actualY)):
    PYPpwd.append(abs(PYPpw[c]-actualY[c]))
    PYPpwd2.append((PYPpw[c]-actualY[c])**2)
    if PYPpwd[c] <= 5:
        PYPpwcount.append(1)
    else:
        PYPpwcount.append(0)
LRYwd = []
LRYwd2 = []
LRYwcount = []
for c in range(0, len(actualY)):
    LRYwd.append(abs(LRYw[c]-actualY[c]))
    LRYwd2.append((LRYw[c]-actualY[c])**2)
    if LRYwd[c] <= 5:
        LRYwcount.append(1)
    else:
        LRYwcount.append(0)
PRYwd = []
PRYwd2 = []
PRYwcount = []
for c in range(0, len(actualY)):
    PRYwd.append(abs(PRYw[c]-actualY[c]))
    PRYwd2.append((PRYw[c]-actualY[c])**2)
    if PRYwd[c] <= 5:
        PRYwcount.append(1)
    else:
        PRYwcount.append(0)
LSYwd = []
LSYwd2 = []
LSYwcount = []
for c in range(0, len(actualY)):
    LSYwd.append(abs(LSYw[c]-actualY[c]))
    LSYwd2.append((LSYw[c]-actualY[c])**2)
    if LSYwd[c] <= 5:
        LSYwcount.append(1)
    else:
        LSYwcount.append(0)
GSYwd = []
GSYwd2 = []
GSYwcount = []
for c in range(0, len(actualY)):
    GSYwd.append(abs(GSYw[c]-actualY[c]))
    GSYwd2.append((GSYw[c]-actualY[c])**2)
    if GSYwd[c] <= 5:
        GSYwcount.append(1)
    else:
        GSYwcount.append(0)
SSYwd = []
SSYwd2 = []
SSYwcount = []
for c in range(0, len(actualY)):
    SSYwd.append(abs(SSYw[c]-actualY[c]))
    SSYwd2.append((SSYw[c]-actualY[c])**2)
    if SSYwd[c] <= 5:
        SSYwcount.append(1)
    else:
        SSYwcount.append(0)
PEDwd = []
PEDwd2 = []
PEDwcount = []
for c in range(0, len(actualY)):
    PEDwd.append(abs(Pedw[c]-actualY[c]))
    PEDwd2.append((Pedw[c]-actualY[c])**2)
    if PEDwd[c] <= 5:
        PEDwcount.append(1)
    else:
        PEDwcount.append(0)
PSEwd = []
PSEwd2 = []
PSEwcount = []
for c in range(0, len(actualY)):
    PSEwd.append(abs(Psew[c]-actualY[c]))
    PSEwd2.append((Psew[c]-actualY[c])**2)
    if PSEwd[c] <= 5:
        PSEwcount.append(1)
    else:
        PSEwcount.append(0)
PSDwd = []
PSDwd2 = []
PSDwcount = []
for c in range(0, len(actualY)):
    PSDwd.append(abs(Psdw[c]-actualY[c]))
    PSDwd2.append((Psdw[c]-actualY[c])**2)
    if PSDwd[c] <= 5:
        PSDwcount.append(1)
    else:
        PSDwcount.append(0)
P1wd = []
P1wd2 = []
P1wcount = []
for c in range(0, len(actualY)):
    P1wd.append(abs(P1w[c]-actualY[c]))
    P1wd2.append((P1w[c]-actualY[c])**2)
    if P1wd[c] <= 5:
        P1wcount.append(1)
    else:
        P1wcount.append(0)
    
pmYd = []
pmYd2 = []
pmYcount = []
for c in range(0, len(actualY)):
    pmYd.append(abs(actual9[c]-actualY[c]))
    pmYd2.append((actual9[c]-actualY[c])**2)
    if pmYd[c] <= 5:
        pmYcount.append(1)
    else:
        pmYcount.append(0)


# In[31]:

#Create DataFrame    
outputDF = pd.DataFrame()

outputDF["antibiotic"] = antibiotic
outputDF["bacteria"] = bacteria
#outputDF["location"] = location
outputDF["tY"] = tY
outputDF["y9"] = y9
outputDF["y1"] = y1

outputDF["r"] = r
outputDF["rY"] = rY
outputDF["ry9"] = ry9
outputDF["ry8"] = ry8

outputDF["actualY"] = actualY
outputDF["actual9"] = actual9
outputDF["actual8"] = actual8
outputDF["sdY"] = sdY
outputDF["sd9"] = sd9
outputDF["sd8"] = sd8

outputDF["RegLinY"] = RegLinY
outputDF["RegPolyY"] = RegPolyY
outputDF["SVRlinY"] = SVRlinY
outputDF["SVRsigY"] = SVRsigY
outputDF["SVRrbfY"] = SVRrbfY

outputDF["PSDw"] = Psdw

#outputDF["PYPER"] = PYPER
#outputDF["PYPERmethod"] = PYPERmethod

outputDF["pmYd"] = pmYd

outputDF["PYPd"] = PYPd
outputDF["PYPAd"] = PYPAd
outputDF["PYPrd"] = PYPrd
outputDF["PYPpd"] = PYPpd
outputDF["LRYd"] = LRYd
outputDF["PRYd"] = PRYd
outputDF["LSYd"] = LSYd
outputDF["GSYd"] = GSYd
outputDF["SSYd"] = SSYd
outputDF["PEDd"] = PEDd
outputDF["PSDd"] = PSDd
outputDF["PSEd"] = PSEd
outputDF["P1d"] = P1d

outputDF["PYPwd"] = PYPwd
outputDF["PYPawd"] = PYPawd
outputDF["PYPrwd"] = PYPrwd
outputDF["PYPpwd"] = PYPpwd
outputDF["LRYwd"] = LRYwd
outputDF["PRYwd"] = PRYwd
outputDF["LSYwd"] = LSYwd
outputDF["GSYwd"] = GSYwd
outputDF["SSYwd"] = SSYwd
outputDF["PEDwd"] = PEDwd
outputDF["PSDwd"] = PSDwd
outputDF["PSEwd"] = PSEwd
outputDF["P1wd"] = P1wd

outputDF["pmYd2"] = pmYd2

outputDF["PYPd2"] = PYPd2
outputDF["PYPAd2"] = PYPAd2
outputDF["PYPrd2"] = PYPrd2
outputDF["PYPpd2"] = PYPpd2
outputDF["LRYd2"] = LRYd2
outputDF["PRYd2"] = PRYd2
outputDF["LSYd2"] = LSYd2
outputDF["GSYd2"] = GSYd2
outputDF["SSYd2"] = SSYd2
outputDF["PEDd2"] = PEDd2
outputDF["PSDd2"] = PSDd2
outputDF["PSEd2"] = PSEd2
outputDF["P1d2"] = P1d2

outputDF["PYPwd2"] = PYPwd2
outputDF["PYPawd2"] = PYPawd2
outputDF["PYPrwd2"] = PYPrwd2
outputDF["PYPpwd2"] = PYPpwd2
outputDF["LRYwd2"] = LRYwd2
outputDF["PRYwd2"] = PRYwd2
outputDF["LSYwd2"] = LSYwd2
outputDF["GSYwd2"] = GSYwd2
outputDF["SSYwd2"] = SSYwd2
outputDF["PEDwd2"] = PEDwd2
outputDF["PSDwd2"] = PSDwd2
outputDF["PSEwd2"] = PSEwd2
outputDF["P1wd2"] = P1wd2

outputDF["pmYcount"] = pmYcount

outputDF["PYPcount"] = PYPcount
outputDF["PYPAcount"] = PYPAcount
outputDF["PYPrcount"] = PYPrcount
outputDF["PYPpcount"] = PYPpcount
outputDF["LRYcount"] = LRYcount
outputDF["PRYcount"] = PRYcount
outputDF["LSYcount"] = LSYcount
outputDF["GSYcount"] = GSYcount
outputDF["SSYcount"] = SSYcount
outputDF["PEDcount"] = PEDcount
outputDF["PSDcount"] = PSDcount
outputDF["PSEcount"] = PSEcount
outputDF["P1count"] = P1count

outputDF["PYPwcount"] = PYPwcount
outputDF["PYPawcount"] = PYPawcount
outputDF["PYPrwcount"] = PYPrwcount
outputDF["PYPpwcount"] = PYPpwcount
outputDF["LRYwcount"] = LRYwcount
outputDF["PRYwcount"] = PRYwcount
outputDF["LSYwcount"] = LSYwcount
outputDF["GSYwcount"] = GSYwcount
outputDF["SSYwcount"] = SSYwcount
outputDF["PEDwcount"] = PEDwcount
outputDF["PSDwcount"] = PSDwcount
outputDF["PSEwcount"] = PSEwcount
outputDF["P1wcount"] = P1wcount

outputDF.head()


# In[32]:

#Before Outlier Removal Stats
outputDF14 = outputDF[outputDF.tY==2014]
outputDF15 = outputDF[outputDF.tY==2015]
outputDF16 = outputDF[outputDF.tY==2016]

Method = ["All", "y14", "y15", "y16", "forecast", "samples", "priorReports", "RecentReports", "nYears", "PreviousValue", "PYPER", "PYPER-ranked", "PYPER-all", "PYPER-withPrevious", "LinearRegression", "PolynomialRegression", "LinearSVR", "GaussianSVR", "SigmoidSVR", "PYPER-errorPropegation", "PYPER-standardDeviation", "PYPER-standardError", "PYPER-threshold1", "w PYPER", "w PYPER-ranked", "w PYPER-all", "w PYPER-withPrevious", "w LinearRegression", "w PolynomialRegression", "w LinearSVR", "w GaussianSVR", "w SigmoidSVR", "w PYPER-errorPropegation", "w PYPER-standardDeviation", "w PYPER-standardError", "w PYPER-threshold1"]
t = outputDF.shape[0]
Avg = [outputDF.shape[0], outputDF14.shape[0], outputDF15.shape[0], outputDF16.shape[0], yearsAhead, samples, priorReports, reportsY, str(nYears) + "-" + str(totalYears), sum(outputDF.pmYd)/t, sum(outputDF.PYPd)/t, sum(outputDF.PYPrd)/t, sum(outputDF.PYPAd)/t, sum(outputDF.PYPpd)/t, sum(outputDF.LRYd)/t, sum(outputDF.PRYd)/t, sum(outputDF.LSYd)/t, sum(outputDF.GSYd)/t, sum(outputDF.SSYd)/t, sum(outputDF.PEDd)/t, sum(outputDF.PSDd)/t, sum(outputDF.PSEd)/t, sum(outputDF.P1d)/t, sum(outputDF.PYPwd)/t, sum(outputDF.PYPrwd)/t, sum(outputDF.PYPawd)/t, sum(outputDF.PYPpwd)/t, sum(outputDF.LRYwd)/t, sum(outputDF.PRYwd)/t, sum(outputDF.LSYwd)/t, sum(outputDF.GSYwd)/t, sum(outputDF.SSYwd)/t, sum(outputDF.PEDwd)/t, sum(outputDF.PSDwd)/t, sum(outputDF.PSEwd)/t, sum(outputDF.P1wd)/t]
RMSE = [outputDF.shape[0], outputDF14.shape[0], outputDF15.shape[0], outputDF16.shape[0], yearsAhead, samples, priorReports, reportsY, str(nYears) + "-" + str(totalYears), (sum(outputDF.pmYd2)/t)**(1/2), (sum(outputDF.PYPd2)/t)**(1/2), (sum(outputDF.PYPrd2)/t)**(1/2), (sum(outputDF.PYPAd2)/t)**(1/2), (sum(outputDF.PYPpd2)/t)**(1/2), (sum(outputDF.LRYd2)/t)**(1/2), (sum(outputDF.PRYd2)/t)**(1/2), (sum(outputDF.LSYd2)/t)**(1/2), (sum(outputDF.GSYd2)/t)**(1/2), (sum(outputDF.SSYd2)/t)**(1/2), (sum(outputDF.PEDd2)/t)**(1/2), (sum(outputDF.PSDd2)/t)**(1/2), (sum(outputDF.PSEd2)/t)**(1/2), (sum(outputDF.P1d2)/t)**(1/2), (sum(outputDF.PYPwd2)/t)**(1/2), (sum(outputDF.PYPrwd2)/t)**(1/2), (sum(outputDF.PYPawd2)/t)**(1/2), (sum(outputDF.PYPpwd2)/t)**(1/2), (sum(outputDF.LRYwd2)/t)**(1/2), (sum(outputDF.PRYwd2)/t)**(1/2), (sum(outputDF.LSYwd2)/t)**(1/2), (sum(outputDF.GSYwd2)/t)**(1/2), (sum(outputDF.SSYwd2)/t)**(1/2), (sum(outputDF.PEDwd2)/t)**(1/2), (sum(outputDF.PSDwd2)/t)**(1/2), (sum(outputDF.PSEwd2)/t)**(1/2), (sum(outputDF.P1wd2)/t)**(1/2)]
pct5 = [outputDF.shape[0], outputDF14.shape[0], outputDF15.shape[0], outputDF16.shape[0], yearsAhead, samples, priorReports, reportsY, str(nYears) + "-" + str(totalYears),100*sum(outputDF.pmYcount)/t, 100*sum(outputDF.PYPcount)/t, 100*sum(outputDF.PYPrcount)/t, 100*sum(outputDF.PYPAcount)/t, 100*sum(outputDF.PYPpcount)/t, 100*sum(outputDF.LRYcount)/t, 100*sum(outputDF.PRYcount)/t, 100*sum(outputDF.LSYcount)/t, 100*sum(outputDF.GSYcount)/t, 100*sum(outputDF.SSYcount)/t, 100*sum(outputDF.PEDcount)/t, 100*sum(outputDF.PSDcount)/t, 100*sum(outputDF.PSEcount)/t, 100*sum(outputDF.P1count)/t, 100*sum(outputDF.PYPwcount)/t, 100*sum(outputDF.PYPrwcount)/t, 100*sum(outputDF.PYPawcount)/t, 100*sum(outputDF.PYPpwcount)/t, 100*sum(outputDF.LRYwcount)/t, 100*sum(outputDF.PRYwcount)/t, 100*sum(outputDF.LSYwcount)/t, 100*sum(outputDF.GSYwcount)/t, 100*sum(outputDF.SSYwcount)/t, 100*sum(outputDF.PEDwcount)/t, 100*sum(outputDF.PSDwcount)/t, 100*sum(outputDF.PSEwcount)/t, 100*sum(outputDF.P1wcount)/t]
metricDF = pd.DataFrame()
metricDF["Method"] = Method
metricDF["MAE"] = Avg
metricDF["RMSE"] = RMSE
metricDF["pct5"] = pct5
print(metricDF)


# In[33]:

from scipy import stats
print(sum(PSDwd)/t)
print(sum(LRYwd)/t)
stats.ttest_ind(PSDwd,LRYwd)


# In[63]:

#import matplotlib.pyplot as plt
#plt.hist(sd9, color = (125/255, 0, 0, .5))
#plt.xlabel("Standard Deviation")
#plt.ylabel("Frequency")
#plt.title("Distribution of Standard Deviation for AP Dataset")
#plt.savefig("sdAP2.pdf")


# In[67]:

print(names)


# In[117]:

bacteriaDF = outputDF[(outputDF.bacteria == "E. coli") & (outputDF.tY == 2016)]
plt.figure()
plt.figure(figsize=(9,4.5))
names = bacteriaDF.antibiotic
numbers = []
for n in range(0, len(names)):
    numbers.append(2*n)
plt.scatter(numbers, bacteriaDF.LRYwd, s = 50, label = "Linear Reg", color = (125/255, 0, 0), marker = 'o', alpha = 0.75)
#plt.scatter(numbers, bacteriaDF.PR14p15d, label = "Polynomial Reg", color = "green", marker = '^')
plt.scatter(numbers, bacteriaDF.LSYwd, s = 50, label = "Linear SVR", color = (0, 125/255, 0), marker = 's', alpha = 0.75)
plt.scatter(numbers, bacteriaDF.GSYwd, s = 50, label = "Gaussian SVR", color = (0, 0, 125/255), marker = '^', alpha = 0.75)
plt.legend(loc = 2, fontsize = 12, scatterpoints = 1)
plt.xticks(numbers, names, rotation = 'vertical', fontsize=6)
plt.title("Escherichia coli", style = 'italic', fontsize = 16) 
#plt.suptitle("Absolute Error of", fontsize = 15)  #Using Data From 2006-2014 to Predict 2015", fontsize = 15)
axes = plt.gca()
axes.set_ylim([0,12])
axes.set_ylabel("Absolute Error")
axes.set_xlabel("Antibiotic")
axes.set_xlim([-1,25])
#plt.subplots_adjust(bottom=0.15)
#plt.savefig('AE1.eps', bbox_inches='tight')

plt.show()


# In[152]:

bacteriaDF = outputDF[(outputDF.bacteria == "E. coli") & (outputDF.tY == 2015)]
plt.figure()
plt.figure(figsize=(9,4.5))
names = bacteriaDF.antibiotic
numbers = []
for n in range(0, len(names)):
    numbers.append(2*n)
#plt.scatter(numbers, bacteriaDF.PYPwd, s = 50, label = "PYPER", color = "purple", marker = '8', alpha = 0.75)
#plt.scatter(numbers, bacteriaDF.PR14p15d, label = "Polynomial Reg", color = "green", marker = '^')
#plt.scatter(numbers, bacteriaDF.LSYwd, s = 50, label = "Linear SVR", color = (0, 125/255, 0), marker = '*', alpha = 0.75)
plt.scatter(numbers, bacteriaDF.LRYwd, s = 50, label = "Linear Reg", color = (125/255, 0, 0), marker = 'o', alpha = 0.25)
#plt.scatter(numbers, bacteriaDF.PR14p15d, label = "Polynomial Reg", color = "green", marker = '^')
plt.scatter(numbers, bacteriaDF.LSYwd, s = 50, label = "Linear SVR", color = (0, 125/255, 0), marker = 's', alpha = 0.25)
plt.scatter(numbers, bacteriaDF.GSYwd, s = 50, label = "Gaussian SVR", color = (0, 0, 125/255), marker = '^', alpha = 0.25)
plt.scatter(numbers, bacteriaDF.PSDwd, s = 75, label = "PYPERed", color = "black", marker = '*', alpha = 0.9)
plt.legend(loc = 2, fontsize = 12, scatterpoints = 1)
plt.xticks(numbers, names, rotation = 'vertical', fontsize=6)
plt.title("Escherichia coli", style = 'italic', fontsize = 16) 
#plt.suptitle("Absolute Error of", fontsize = 15)  #Using Data From 2006-2014 to Predict 2015", fontsize = 15)
axes = plt.gca()
axes.set_ylim([0,12])
axes.set_ylabel("Absolute Error")
axes.set_xlabel("Antibiotic")
axes.set_xlim([-1,25])
#plt.subplots_adjust(bottom=0.15)
#plt.savefig('AE2.eps', bbox_inches='tight')

plt.show()


# In[140]:

print(set(outputDF.antibiotic))


# In[149]:

mini = outputDF[outputDF.antibiotic=="CEFUROXIME"]
print(sum(mini.PSDwd)/len(mini.PSDwd))


# In[176]:

ceph = df[(df.component == "CEPHALOTHIN") & (df.organism == "Klebsiella oxytoca")]
ceph=ceph.set_index(np.arange(0,ceph.shape[0]))
x = []
y = []
z = []
for i in range(0, len(ceph.organism)):
    x.append(ceph["Report Year"][i])
    y.append(ceph["Indicator Value (Pct)"][i])
    z.append(ceph["Total Tests (by organism)"][i]/75)
plt.scatter(x, y, z, color = (125/255, 0, 0, .5))
plt.show()


# In[194]:

x = [71.7179689,70.85208711,70.75987635,67.61112481,59.95711194,58.06975061,54.32756714,52.20230179,57.09184329,55.32391264,53.46066171,39.39843449,43.35717155,33.19733925,41.13060429]
x2 = [70.6194895, 69.99048913, 78.97713098, 67.7122449, 64.58293839, 72.40718563, 66.49609375, 65.18624642, 65.52341137, 65.26277372, 64.30769231, 47.09459459, 67.01149425]

y = []
for year in range(2002, 2015):
    y.append(year)
plt.scatter(y, x2, color = (125/255, 0, 0, .5))
plt.show()


# In[186]:

df39 = pd.DataFrame()
df39["antibiotic"] = outputDF["antibiotic"]
df39["bacteria"] = outputDF["bacteria"]
df39["error"] = outputDF["PSDwd"]
df39["year"] = outputDF["tY"]
df39 = df39.sort_values(by='error')
df39


# In[207]:

df39 = pd.DataFrame()
df39["antibiotic"] = outputDF["antibiotic"]
df39["bacteria"] = outputDF["bacteria"]
df39["error"] = outputDF["PSDwd"]
df39["year"] = outputDF["tY"]
df39["prediction"] = outputDF["PSDw"]
df39 = df39.sort_values(by='error')
df39


# In[193]:

print(len(x1))
print(len(y1))


# In[233]:

x1 = [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015]
y1 = [70.6194895, 69.99048913, 78.97713098, 67.7122449, 64.58293839, 72.40718563, 66.49609375, 65.18624642, 65.52341137, 65.26277372, 64.30769231, 47.09459459, 67.01149425, 43]
z1 = [6.052419544,1.462636255,10.32567101,1.220991254,0.774684165,0.618250634,5.600429947,1.301328214,4.743250842,9.403182257,17.77682046,2.073256912,8.238493679, 0]
sd = [14,15,13,15,8,7,6,6,9,8,4,5,2, 1]
plt.figure()
plt.plot(2015, 67.667, color = "black", marker = "*")
plt.plot(2015, 54.125, color = "black", marker = "^")
plt.scatter(x1,y1, s=z1, color = (125/255, 0, 0, .5))
plt.errorbar(x1,y1,yerr=sd, color = (125/255, 0, 0, .5))
plt.title("2015 PYPERed Predictions") 
axes = plt.gca()
axes.set_ylim([0,100])
axes.set_xlim([2001,2017])
axes.set_ylabel("Susceptibility Percent")
axes.set_xlabel("Year")
p1s = plt.scatter([],[], color = (125/255, 0, 0, .5))
p2s = plt.scatter([],[], color = "black", marker = "^")
p3s = plt.scatter([],[], color = "black", marker = "*")
plt.legend([p1s, p2s, p3s], ("Actual Mean Susceptibility", "Prediction with 3-9 years of data", "Prediction with 6-12 years of data"), scatterpoints=1, title = "Number of\n Reports", frameon=False, fontsize = 9, loc=3)
#plt.savefig('dataCeph.eps', bbox_inches='tight')
plt.show()


# In[232]:

x1 = [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
y1 = [96.17084247, 94.37568306, 91.93107574,87.87610631,85.75370588,85.25451127,83.04064371,81.49983583,81.49453057,81.09801782,81.0789069,80.04235386,79.84869926,78.47649695,79.39738182]
z1 = [38,52,59,54,46,48,52,44,46,50,48,50,47,49,49]
sd = [3.564696703, 0.534600604, 4.217797464,3.953429445,2.330974231,7.255187644,7.573559101,2.839417508,6.139773224,0.014757438,0.782308367,2.416611393,2.613570965,0.663394587,0.480372715]
plt.figure()
plt.scatter(x1,y1, s=z1, color = (125/255, 0, 0, .5))
plt.errorbar(x1,y1,yerr=sd, color = (125/255, 0, 0, .5))
plt.title("Time Series View") 
axes = plt.gca()
axes.set_ylim([0,100])
axes.set_xlim([2001,2017])
axes.set_ylabel("Susceptibility Percent")
axes.set_xlabel("Year")
p1s = plt.scatter([],[], s=z1[0], color = (125/255, 0, 0, .5))
p2s = plt.scatter([],[], s=z1[9], color = (125/255, 0, 0, .5))
p3s = plt.scatter([],[], s=z1[2], color = (125/255, 0, 0, .5))
plt.legend([p1s, p2s, p3s], ("40", "50", "60"), scatterpoints=1, title = "Number of\n Reports", frameon=False, fontsize = 9, loc=3)
#plt.savefig('data2.pdf', bbox_inches='tight')
plt.show()


# In[196]:

ceph = ceph[ceph["Report Year"] == 2015]
ceph


# In[239]:

x1 = [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
y1 = [99.38352273,99.5994832,99.75061576,99.71278983,99.52170139,99.64620631,99.54642857,99.7062635,99.38359202,97.4283552,89.23267327,85.32108027,86.35374771,88.38203191,84.13628763]
z1 = [26,34,40,35,28,29,29,24,28,33,32,30,25,26,24]
sd = [0.820801296,0.585605992,0.320363761,0.293899016,0.401560415,0.448494122,2.089335388,0.460104613,0.527849635,1.556422386,9.793431061,44.62387967,31.85170473,33.81350245,26.2621709]
plt.figure()
plt.scatter(x1,y1, s=z1, color = (125/255, 0, 0, .5))
plt.errorbar(x1,y1,yerr=sd, color = (125/255, 0, 0, .5))
plt.title("CLSI Change") 
axes = plt.gca()
axes.set_ylim([0,100])
axes.set_xlim([2001,2017])
axes.set_ylabel("Susceptibility Percent")
axes.set_xlabel("Year")
p1s = plt.scatter([],[], s=z1[0], color = (125/255, 0, 0, .5))
p2s = plt.scatter([],[], s=z1[5], color = (125/255, 0, 0, .5))
p3s = plt.scatter([],[], s=z1[1], color = (125/255, 0, 0, .5))
p4s = plt.scatter([],[], s=z1[2], color = (125/255, 0, 0, .5))
plt.legend([p1s, p2s, p3s, p4s], ("25", "30", "35", "40"), scatterpoints=1, title = "Number of\n Reports", frameon=False, fontsize = 9, loc=3)
plt.savefig('CLSI.eps', bbox_inches='tight')
#plt.show()


# In[ ]:



