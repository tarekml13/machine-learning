import matplotlib.pyplot as plt
import numpy as np
import math
import random

def w0w1(xn, tn):
    xn = np.array(xn)
    tn = np.array(tn)
    xn_mean = np.mean(xn)
    tn_mean = np.mean(tn)
    xn_tn_mean = np.mean(xn*tn)
    xn2_mean = np.mean(xn**2)
    w1 = (xn_tn_mean - xn_mean*tn_mean)/(xn2_mean - xn_mean**2)
    w0 = tn_mean - w1*xn_mean
    return w0, w1

def X_matrix(xn):
    x = np.array(xn)
    return np.array([np.ones(len(x)),x,np.sin((x-2660)/4.3)]).T

def w(xn, tn):
    xn = np.array(xn)
    tn = np.array(tn)
    X = X_matrix(xn)
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),tn)

def order_fun(xn, tn, n):
    xn = np.array(xn)
    tn = np.array(tn)
    x0 = xn[0]
    xn = (xn - x0)/4
    X = [np.ones(len(xn))] 
    for i in range(1,n+1):
        X.append(xn**i)
    X = np.array(X).transpose()
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),tn)
    O_loss = loss(w,X,tn)
    print(
        f"the average loss - Order {n} : {O_loss / len(xn)}"
    )
    plt.plot(xn,tn,'ro',label = "men")
    xplot = np.linspace(min(xn),max(xn),50)
    yplot = 0
    for i in range(n+1):
        yplot = yplot + w[i]*xplot**i
    if (n==1):
        titleOrder = "1st order" 
    elif (n==2):
        titleOrder = "2nd order"
    else:
        titleOrder = f"{n}th order"
    plt.plot(xplot,yplot,label = titleOrder)
    plt.title(titleOrder)
    plt.xlabel("years")
    plt.ylabel("seconds")
    plt.legend(loc="upper right")
    plt.show()
    return O_loss

#Function to calculate the error (squared loss)
def error(w0,w1,xn,tn):
    return sum((tn[i] - (w0 + w1*xn[i]))**2 for i in range(len(xn)))

def loss (w,X,t):
    return np.dot((t-np.dot(X,w)).T,(t-np.dot(X,w)))
#Exo 01
xn = []
tn = []
for i in range(5):
    xn.append(random.randint(2010, 2023))
    tn.append(math.sin(2*math.pi*xn[i]) + random.gauss(0, 0.3))
    print(f"x{i + 1} = {xn[i]}, t{i + 1} = {tn[i]}")

w0,w1 = w0w1(xn,tn)
print(f"w0 = {w0}, w1 = {w1}")
print(f"Error = {error(w0, w1, xn, tn)}")
#Exo 02
xn_Women = [1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008]
tn_Women = [12.20, 11.90, 11.50, 11.90, 11.50, 11.50, 11.00, 11.40, 11.00, 11.07, 11.08, 11.06, 10.97, 10.54, 10.82, 10.94, 11.12, 10.93, 10.78]
w0_W,w1_W = w0w1(xn_Women,tn_Women)
if (w1_W < 0):
    print(
        f"linear model that minimizes the squared loss for women is t = {w0_W} {w1_W} x"
    )
else:
    print(
        f"linear model that minimizes the squared loss is t = {w0_W} + {w1_W} x"
    )
print(
    f"womans winning time at the 2012 Olympic games is : {w0_W + w1_W * 2012} seconds"
)
print(
    f"womans winning time at the 2016 Olympic games is : {w0_W + w1_W * 2016} seconds"
)

#Exo 03
xn_Men = [1896, 1900, 1904, 1906, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008]
tn_Men = [12.00, 11.00, 11.00, 11.20, 10.80, 10.80, 10.80, 10.60, 10.80, 10.30, 10.30, 10.30, 10.40, 10.50, 10.20, 10.00, 9.95, 10.14, 10.06, 10.25, 9.99, 9.92, 9.96, 9.84, 9.87, 9.85,9.69]
w0_M,w1_M = w0w1(xn_Men,tn_Men)
print(
    f"the Olympic games when it is predicted for women to run a faster winning time than men is : {4 * ((w0_M - w0_W) / (w1_W - w1_M) // 4 + 1)}"
)
print(
    f"the predicted winning times is : {w0_W + w1_W * ((w0_M - w0_W) / (w1_W - w1_M))}"
)
plt.plot(xn_Women,tn_Women,'ro',label="women")
plt.plot([1928,2600],[w0_W + w1_W*1928,w0_W + w1_W*2600])
plt.plot(xn_Men,tn_Men,'bo',label="man")
plt.plot([1928,2600],[w0_M + w1_M*1928,w0_M + w1_M*2600])
plt.title("Olympic men's and women's 100m data")
plt.xlabel("years")
plt.ylabel("seconds")
plt.legend(loc="upper right")
plt.show()

#Exo 04
X_M = X_matrix(xn_Men)
w_M = w(xn_Men,tn_Men)
X_W = X_matrix(xn_Women)
w_W = w(xn_Women,tn_Women)
loss_M = loss(w_M,X_M,tn_Men)
print(f"loss for Mens 100m data : {loss_M}")
loss_W = loss(w_W,X_W,tn_Women)
print(f"loss for Womens 100m data: {loss_W}")
plt.plot(xn_Women,tn_Women,'ro',label="women")
plt.plot(xn_Men,tn_Men,"bo",label="man")
xplot = np.linspace(1928,2016,50)
plt.plot(xplot,w_W[0]+w_W[1]*xplot+w_W[2]*np.sin((xplot-2660)/4.3))
plt.plot(xplot,w_M[0]+w_M[1]*xplot+w_M[2]*np.sin((xplot-2660)/4.3))
plt.legend(loc="upper right")
plt.show()

#orders
lossList = []
for i in range(8):
    lossOrder=order_fun(xn_Men,tn_Men,i+1)
    lossList.append(lossOrder)
plt.plot([1,2,3,4,5,6,7,8],lossList)
plt.xlabel("order")
plt.ylabel("loss")
plt.show()

#cross-validation
xn_M_training = xn_Men[:14]
xn_M_validation = xn_Men[14:]
tn_M_training = tn_Men[:14]
tn_M_validation = tn_Men[14:]
#weights 
w_M_training = w(xn_M_training,tn_M_training)
#loss
loss_M_validation = loss(w_M_training,X_matrix(xn_M_validation),tn_M_validation)
print(f"the average loss with validation data set is : {loss_M_validation/len(xn_M_validation)}")
#plot 
plt.plot(xn_M_training,tn_M_training,'.',label = "training data")
plt.plot(xn_M_validation,tn_M_validation,'.',label = "validation data")
xplot = np.linspace(1896,2008,50)
plt.plot(xplot,w_M_training[0]+w_M_training[1]*xplot)
plt.title("cross-validation")
plt.xlabel("years")
plt.ylabel("seconds")
plt.legend(loc="upper right")
plt.show()
