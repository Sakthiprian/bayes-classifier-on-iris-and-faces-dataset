import numpy as np
import pandas as pd
import math
import helpers


data = pd.read_csv("iris - iris.csv")

#split dataset into each class

versicolor = data.loc[data["Species"] == "Iris-versicolor"]
setosa = data.loc[data["Species"] == "Iris-setosa"]
virginica = data.loc[data["Species"] == "Iris-virginica"]

versicolordrop = versicolor.drop(["Species", "Id"], axis=1)
setosadrop = setosa.drop(["Species", "Id"], axis=1)
virginicadrop = virginica.drop(["Species", "Id"], axis=1)

#randomly select a sample from each class
test = pd.concat([setosa.tail(10), versicolor.tail(10), virginica.tail(10)])

#drop the testset from trainset
train = data.drop(test.index, errors='ignore')

totalSamples=150
sampleCnt=50

apriori_versi =apriori_vir =apriori_set =helpers.apriori(sampleCnt,totalSamples)

print(apriori_set,apriori_versi,apriori_vir)

covMatSet=setosadrop.cov(numeric_only=True)
covMatVer=versicolordrop.cov(numeric_only=True)
covMatVir=virginicadrop.cov(numeric_only=True)

meanvectset=setosadrop.mean(numeric_only=True)
meanvectver=versicolordrop.mean(numeric_only=True)
meanvectvir=virginicadrop.mean(numeric_only=True)

for index, row in test.iterrows():
    testvect = row[["PetalWidthCm", "PetalLengthCm", "SepalWidthCm", "SepalLengthCm"]].values
    ccpdf_setosa=helpers.ccpdf(testvect,meanvectset,covMatSet)
    ccpdf_versi=helpers.ccpdf(testvect,meanvectver,covMatVer)
    ccpdf_virginica=helpers.ccpdf(testvect,meanvectvir,covMatVir) 
    print(ccpdf_setosa,ccpdf_versi,ccpdf_virginica)
    lst=[helpers.Belongingness(ccpdf_setosa,apriori_set),helpers.Belongingness(ccpdf_versi,apriori_versi),helpers.Belongingness(ccpdf_virginica,apriori_vir)]
    pred=helpers.Compare(lst)
    print(pred)




