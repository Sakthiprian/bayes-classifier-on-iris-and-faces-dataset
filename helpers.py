import math
import numpy as np

epsilon = math.pow(10,-40)
def apriori(n_wi,tot):
    return  n_wi/tot

def ccpdf(covMat, dimCount, meanVect, X):
    invCovMat = np.linalg.inv(covMat)
    detCovMat = np.linalg.det(covMat)
    diff= X - meanVect
    const_part=1/(((2*np.pi)**(dimCount/2))*detCovMat**0.5)
    exp_part=np.exp(-0.5*(np.matmul(np.matmul(np.transpose(diff),invCovMat),diff)))
    likelihood = const_part * exp_part
    return likelihood
"""
def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    sigma += np.eye(size) * epsilon

    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = np.linalg.inv(sigma)
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")"""

def Belongingness(ccpdf,apriori):
    return ccpdf*apriori

def Compare(lst,set):
    if set=="iris":
        ind=lst.index(max(lst))
        if ind==0:
            return "Setosa"
        elif ind==1:
            return "Versicolor"
        elif ind==2:
            return "Virginica"
    else:
        ind=lst.index(max(lst))
        if ind==0:
            return "male"
        elif ind==1:
            return "female"
