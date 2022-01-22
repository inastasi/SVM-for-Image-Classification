import Project_2_dataExtraction as pde
import part_4 as fun
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import time

'''
Taking data
'''
# Setting a seed
np.random.seed ( 1861402 )

# Read input data
xLabel2, yLabel2, xLabel4, yLabel4, xLabel6, yLabel6 = pde.returnData ()
yLabel2 = np.array ( [2] * 1000 )
yLabel4 = np.array ( [4] * 1000 )
yLabel6 = np.array ( [6] * 1000 )

label2_data = np.append ( xLabel2, yLabel2.reshape ( 1000, 1 ), axis=1 )
label4_data = np.append ( xLabel4, yLabel4.reshape ( 1000, 1 ), axis=1 )
label6_data = np.append ( xLabel6, yLabel6.reshape ( 1000, 1 ), axis=1 )


all_data = np.append ( label2_data,label4_data, axis=0)
all_data = np.append ( all_data,label6_data, axis=0)
 
'''
Setting hyper-parameters
'''
gamma = 0.001
C = 2.1
tollerance = 0.00001
start_time = time.time ()

'''
Defining strategy 

ONE VS ONE Strategy 

    1) Training three classifiers 

    Classifier 1 - label2 VS label4
    Classifier 2 - label2 VS label6
    Classifier 3 - label4 VS label6
    
    label2 = coat
    label4 = pullover
    label6 = shirt

2) Combining results of the three classifiers 
'''


all_combos = [(2.,4.),(2.,6.),(4.,6.)]
all_classifier_combos=[]
X = all_data[:, :-1]
Y = all_data[:, -1]  # .reshape((-1,1))
X_train, X_test, y_train, y_test = train_test_split ( X, Y, test_size=0.2 )

"""
Scaling data
"""
scaler = preprocessing.StandardScaler ()
X_train = scaler.fit_transform ( X_train )
X_test = scaler.transform ( X_test )
train_data = np.append ( X_train, y_train.reshape ( len ( y_train ), 1 ), axis=1 )
test_data = np.append ( X_test, y_test.reshape ( len ( y_test ), 1 ), axis=1 )

"""
Training each of the classifiers separately for the above mentioned pairs
and Testing predictive power
"""
accuracy_train_cl=[]
KKT_cl=[]
fun_evals_cl=[]

for idx,combo in enumerate(all_combos):
    print ( "-------------------------------" )
    print("Combination ",combo)

    ## training data for 2 classes (each combination of class pairs)
    cl_data=train_data[np.where((train_data[:,-1]==combo[0]) | (train_data[:,-1]==combo[1]))]
    X=cl_data[:,:-1]
    Y=cl_data[:,-1].reshape((-1,1))

    classes = np.unique ( Y )
    Y=np.where(Y==classes[0],-1,1) #one classs is -1, other is +1
    
    X_train_cl=X
    y_train_cl=Y
    
    
    ## testing data for 2 classes (each combination of class pairs)    
    test_cl_data=test_data[np.where((test_data[:,-1]==combo[0]) | (test_data[:,-1]==combo[1]))]
    X_t=test_cl_data[:,:-1]
    Y_t=test_cl_data[:,-1].reshape((-1,1))



    Y_t=np.where(Y_t==classes[0],-1,1)#one classs is -1, other is +1
    
    X_test_cl=X_t
    y_test_cl=Y_t

   
   # X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split ( X, Y, test_size=0.2 )

    #training classifier; and finding optimal alpha for each classifier
    C, gamma, alpha, funct_eva, final_obj, y_train_pred, sec, primal_inf, dual_inf, primal_slac, dual_slac,alpha_star, X_train_sv, y_train_sv,b = fun.SVM ( X_train_cl, y_train_cl, C, gamma )
    
    #storing the output of SVM for each classifier
    all_classifier_combos.append((X_train_sv, y_train_sv,alpha_star,b,all_combos[idx]))

    #making predictions and evaluating the classifiers predictive power(time and accuracy)
    y_test_pred = np.sign ( ((np.multiply ( fun.RBF_kernel ( X_train_sv, X_test_cl, gamma ),
                                            np.multiply ( alpha_star, y_train_sv ) )).sum ( axis=0 )).reshape (
        (-1, 1) ) + (np.repeat ( b, len ( X_test_cl ), axis=0 )).reshape ( (-1, 1) ) )

    # Accuracy measures
    err_train, conf_mat_train, accuracy_train = fun.pred_error ( y_train_cl, y_train_pred )
    err_test, conf_mat_test, accuracy_test = fun.pred_error ( y_test_cl, y_test_pred )

 
    # Calculate KKT violation
    R,S = fun.r_s(alpha, y_train_cl, C, tollerance)
    # gradient of the Laplacian
    grad = -np.multiply(fun.dual_grad(fun.Q_mat(X_train_cl,y_train_cl,gamma),alpha),y_train_cl.reshape(-1,1))
    
    m = max(np.take(grad, R))
    M = min(np.take(grad, S)) 
    kkt_violation = m-M  
    
    
    accuracy_train_cl.append(accuracy_train)
    KKT_cl.append(kkt_violation)
    fun_evals_cl.append(funct_eva)
    print ( "C value: %s" % (C) )
    print ( "Gamma value: %s" % (gamma) )
    print ( "Accuracy Rate Training Set: %s" % (accuracy_train) )  
    print ( "Confusion matrix: \n", conf_mat_train )
    print ( "Accuracy Rate Test Set: %s" % (accuracy_test) )
    print ( "Confusion matrix: \n", conf_mat_test )
    print ( "Number of function evaluations: ", funct_eva )
    print ( "Final value of the objective function of the dual problem: ", final_obj )
    print("KKT violation: %s"%(kkt_violation))
    print ( "Time: %s seconds" % (sec) )
   
 

"""
Training each of the classifiers separately for the above mentioned pairs
"""
#real test 
res_matrix_tr=np.zeros((len(y_train),3))#cause 3 is the number of one vs one classifiers
res_matrix=np.zeros((len(y_test),3))#cause 3 is the number of one vs one classifiers

for idx,classifier in enumerate(all_classifier_combos):
             
    X_train_sv, y_train_sv, alpha_star, b, cl_combination = classifier
    y_train_pred = np.sign ( ((np.multiply ( fun.RBF_kernel ( X_train_sv, X_train, gamma ),
                                            np.multiply ( alpha_star, y_train_sv ) )).sum ( axis=0 )).reshape (
       (-1, 1) ) + (np.repeat ( b, len ( X_train), axis=0 )).reshape ( (-1, 1) ) )

    y_test_pred = np.sign ( ((np.multiply ( fun.RBF_kernel ( X_train_sv, X_test, gamma ),
                                            np.multiply ( alpha_star, y_train_sv ) )).sum ( axis=0 )).reshape (
       (-1, 1) ) + (np.repeat ( b, len ( X_test), axis=0 )).reshape ( (-1, 1) ) )

    res_matrix_tr[:,idx]=np.where(y_train_pred==-1,cl_combination[0],cl_combination[1]).reshape(len(y_train_pred),)

    res_matrix[:,idx]=np.where(y_test_pred==-1,cl_combination[0],cl_combination[1]).reshape(len(y_test_pred),)
#print(res_matrix)

res_df_tr = pd.DataFrame({'Cl1': res_matrix_tr[:, 0], 'Cl2': res_matrix_tr[:, 1], 'Cl3': res_matrix_tr[:, 2]})
y_train_pred_final=res_df_tr.mode(axis=1)
y_train_pred_final=y_train_pred_final.loc[:,0].values


res_df = pd.DataFrame({'Cl1': res_matrix[:, 0], 'Cl2': res_matrix[:, 1], 'Cl3': res_matrix[:, 2]})
res_df.to_csv('all_cl.csv',index=False)
y_test_pred_final=res_df.mode(axis=1)
y_test_pred_final=y_test_pred_final.loc[:,0].values




err_tr, conf_mat_tr, accuracy_tr = fun.pred_error ( y_train, y_train_pred_final )
err_test, conf_mat_test, accuracy_test = fun.pred_error ( y_test, y_test_pred_final )


end_time = time.time ()

print("-----------------------------------------------------")
print ( "Classifiers combined " )

# printing results
print("C value: %s"%(C))
print("Gamma value: %s"%(gamma))
print("Accuracy rate training set: %s"%(accuracy_tr))
print("Accuracy rate test set: %s"%(accuracy_test))
print("Confusion matrix: \n", conf_mat_test)
print("Number of function evaluations: ",int(np.sum(fun_evals_cl)))
print("KKT violation: %s"%(np.mean(KKT_cl)))
print("Time: %s seconds"%(end_time-start_time))
print("-----------------END------------------------------")
 
 