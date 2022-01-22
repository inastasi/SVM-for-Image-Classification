import Project_2_dataExtraction as pde
import part_1 as q1_func
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


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

# Prepare data for cross-validation
data_cv = np.append(X_train,y_train.reshape((-1,1)),axis=1)

 

'''
# EACH ROW IS A clothing item; if we reshape it to the matrix we get the pic

plt.imshow(X_train[0].reshape(28,28), interpolation='nearest')
plt.show()
'''


# Scaling data
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#plt.imshow(X_train[0].reshape(28,28), interpolation='nearest')
#plt.show()


# Define parameters needed for cvxopt quadratic convex optimization problem solver

gamma = 0.001
C = 2.1
tollerance = 0.00001

# Call optimization function

C, gamma, alpha, funct_eva, final_obj, y_train_pred, y_test_pred, sec = q1_func.SVM(X_train,y_train,X_test, y_test, C, gamma)

# Calculate accuracy
err_train, conf_mat_train, accuracy_train = q1_func.pred_error(y_train, y_train_pred)
err_test, conf_mat_test, accuracy_test = q1_func.pred_error(y_test, y_test_pred)

# Calculate KKT violation
R,S = q1_func.r_s(alpha, y_train, C, tollerance)
# gradient of the Laplacian
grad = -np.multiply(q1_func.dual_grad(q1_func.Q_mat(X_train,y_train,gamma),alpha),y_train.reshape(-1,1))

m = max(np.take(grad, R))
M = min(np.take(grad, S)) 
kkt_violation = m-M  


# printing results
print("Gamma value: %s"%(gamma))
print("C value: %s"%(C))
print("-----------------------------------------------------")
print("Accuracy rate training set: %s"%(accuracy_train))
#print("Training Error: ",err_train)
print("-----------------------------------------------------")
print("Accuracy rate test set: %s"%(accuracy_test))
#print("Test Error: ",err_test)
print("Confusion matrix test: \n", conf_mat_test)
print("-----------------------------------------------------")
print("Time: %s seconds"%(sec))
print("KKT violation: %s"%(kkt_violation))

 
