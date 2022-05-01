import numpy as np

def normpdf(x,mu,sigma):
    pdf=np.exp(-(x-mu)**2/(2*sigma**2))/(sigma * np.sqrt(2 * np.pi))
    return pdf

X=np.arange(30,41,2)
Y=normpdf(X,35,2)
Y=Y*1000
n=0
for i in range(6):
    n=n+Y[i]
    print(n)
t=164
for i in Y:
    m=Y/n
    k=m*t
print(Y)
print(m)
print(k)

# 3:17:200:200:7:3
#
# 0.0125
# 0.0708
# 0.8333
#

print(164)