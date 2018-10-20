
import numpy as np
import matplotlib.pyplot as plt


def Xi():
    XA=np.array([
        [0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.403, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.593, 0.042, 0],
        [0.719, 0.103, 0]
    ])
    XA1=np.array([
        [0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.403, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1]
    ])
    XA2=np.array([
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.593, 0.042, 0],
        [0.719, 0.103, 0]
    ])
    X1=[]
    X2=[]
    for ar in XA1:
        X1.append([float(ar[0]),float(ar[1])])
    for ar in XA2:
        X2.append([float(ar[0]), float(ar[1])])
    X1=np.array(X1)
    X2=np.array(X2)
    return X1,X2

def Ui(X1,X2):
    U1=np.array([np.mean(X1[:,0]),np.mean(X1[:,1])])
    U2=np.array([np.mean(X2[:,0]),np.mean(X2[:,1])])
    m1=np.shape(X1)[0]
    m2=np.shape(X2)[0]
    sw=np.zeros(shape=(2,2))
    for i in range(m1):
        sw += np.dot((X1[i,:]-U1),(X1[i,:]-U1).transpose())
    for i in range(m2):
        sw += np.dot((X2[i,:]-U2),(X2[i,:]-U2).transpose()) #计算类内散度矩阵
    w= np.dot((U2-U1),sw.transpose())
    return w

def pic(w):
    XA = np.array([
        [0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.403, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.593, 0.042, 0],
        [0.719, 0.103, 0]
    ])
    dataMat=[]
    for ar in XA:
        dataMat.append([ar[0], ar[1]])
    dataMat=np.array(dataMat)
    labelMat=np.array([XA[:,2]]).transpose()
    m,n=np.shape(dataMat)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if labelMat[i,0] == 1:
            xcord1.append(dataMat[i, 0])
            ycord1.append(dataMat[i, 1])
        else:
            xcord2.append(dataMat[i, 0])
            ycord2.append(dataMat[i, 1])
    plt.figure(1)
    ax = plt.subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='+')
    ax.scatter(xcord2, ycord2, s=30, c='green',marker='+')
    x = np.arange(-0.2, 0.8, 0.1)
    y = np.array((-w[0] * x) / w[1])
    plt.sca(ax)
    plt.plot(x, y) 
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


    
x1,x2=Xi()
w=Ui(x1,x2)
print(w)
print(pic(w))
