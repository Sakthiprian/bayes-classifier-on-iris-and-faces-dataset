import helpers
import numpy as np
import pandas as pd

data=pd.read_csv('face feature vectors - face feature vectors.csv',index_col=[0])

male=data.iloc[:400,:]
female=data.iloc[400:800,:]

test=pd.concat([male.head(5),female.head(5)],axis=0)

train=data.drop(test.index)

apriori_female=apriori_male=helpers.apriori(400,800)

meanVectMale=male.mean(numeric_only=True)
meanVectFemale= female.mean(numeric_only=True)

covMatMale=male.cov(numeric_only=True)
covMatFemale=female.cov(numeric_only=True)

for index,row in test.iterrows():
    testVect = row[1:].values
    ccpdf_male=helpers.ccpdf(covMatMale,128,meanVectMale,testVect)
    ccpdf_female=helpers.ccpdf(covMatFemale,128,meanVectFemale,testVect)
    lst=[helpers.Belongingness(ccpdf_male,apriori_male),helpers.Belongingness(ccpdf_female,apriori_female)]
    pred=helpers.Compare(lst,"faces")
    print(pred)