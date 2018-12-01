import numpy as np
from pylab import *

def train_wb(X, y):
    if np.linalg.det(X.T * X) != 0:
        wb = ((X.T.dot(X).I).dot(X.T)).dot(y)
        return wb

def test(x, wb):
    return x.T.dot(wb)

def draw(x, y, wb):
    #画回归直线y = wx+b
    a = np.linspace(0, np.max(x)) #横坐标的取值范围
    b = wb[0] + a * wb[1]
    plot(x, y, '.')
    plot(a, b)
    show()

X=np.mat([[1,2],[2,4],[3,6],[5,6],[6,7]])
y=np.mat([2,4,6,8,10]).T
wb = train_wb(X, y)
draw(X[:, 1], y, wb.tolist())