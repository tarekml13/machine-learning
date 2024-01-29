import numpy as np
#features
age = np.array([19, 20, 21, 22, 23, 24, 25, 26])
weight = np.array([60, 67, 70, 71, 45, 64, 80, 90])
salary = np.array([100, 140, 160, 1040, 1070, 1000, 9100, 6100])

#prediction on what
output = np.array([100, 140, 160, 1040, 1070, 1000, 9100, 6100])

X = np.array([age,weight,salary]).T
w =  np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),output)
print(w)

loss = np.dot((output-np.dot(X,w)).T,(output-np.dot(X,w)))
print(loss)

# make prediction
p = np.array([19,60,100]) 
print(np.dot(p, w))