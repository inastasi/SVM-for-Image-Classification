import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import Project_2_dataExtraction as pde
import part_3 as q3_func

#Setting a seed
np.random.seed(1861402)

# Read input data
xLabel2,yLabel2,xLabel4,yLabel4,xLabel6,yLabel6=pde.returnData()

yLabel2=np.array([1]*1000)
yLabel4=np.array([-1]*1000)
label2_data=np.append(xLabel2,yLabel2.reshape(1000,1),axis=1)
label4_data=np.append(xLabel4,yLabel4.reshape(1000,1),axis=1)


all_data=np.append(label2_data,label4_data,axis=0)
X = all_data[:,:-1]
Y = all_data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

# Scaling data
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Predifined hyper-parameter values

gamma = 0.001
C = 2.1
tollerance = 0.00001
q = 2
maxiter = 100000

final_obj = q3_func.MVP(X_train,y_train,X_test,y_test, gamma, q, tollerance, C, maxiter)
print('Final objective function: ',final_obj)
