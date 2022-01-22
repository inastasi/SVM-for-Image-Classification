# -*- coding: utf-8 -*-
import numpy as np
import sklearn as sk
from cvxopt import solvers
from cvxopt import matrix
import time
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd
np.random.seed(1861402)

def RBF_kernel(X,Y,gamma):
    return sk.metrics.pairwise.rbf_kernel(X, Y, gamma)

def Q_mat(X,Y,gamma):
    K = RBF_kernel(X, X, gamma)
    np.reshape(Y,(len(Y),-1))
    Q1 = np.multiply(Y,K)
    Q = matrix(np.multiply(Y,Q1.T).T)
    #Q = matrix(Q2.T)
    return Q

def dual_grad(Q,alpha):
    return np.dot(Q,alpha)-1
#################################################################################################################################

def SVM(X_train,y_train,X_test, y_test, C, gamma):
    
    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-12
    solvers.options['feastol'] = 1e-12
    
    start_time = time.time()

    # matrix Q
    Q = Q_mat(X_train,y_train,gamma)
    
    # vector p, shape (len(X_train)x1)
    p = matrix(np.repeat(-1,len(X_train)).reshape(len(X_train),1),tc='d')
    
    
    # vector A (1 x len(y_train))
    A = matrix(y_train,(1,len(y_train)))
    # vector b is a scalar
    b = matrix(0,tc='d')
    
    # create the first constraint (- alpha <= 0)
    first_constr = np.diag([-1]*len(y_train))
    first_limit = np.array([0]*len(y_train))
    # create the second constraint (alpha <= C)
    second_constr = np.diag([1]*len(y_train))
    second_limit = np.array([C]*len(y_train))
    
    G = matrix(np.concatenate((first_constr, second_constr)),tc='d')
    h = matrix(np.concatenate((first_limit, second_limit)))
    
    # Solve minimization problem
    sol = solvers.qp(Q,p,G,h,A,b)
    
    # Take alpha from the solution
    alpha = np.array(sol['x'])
    
    # Support vectors are ones corresponding to alpha values greater than 1e-5
    ind = np.where(np.any(alpha>0.00001, axis=1))
    
    X_train_sv = X_train[ind]
    y_train_sv = y_train[ind]
    y_train_sv = (y_train_sv.T).reshape((-1,1))
    
    alpha_star = alpha[ind]
    
    funct_eva = sol["iterations"]
    final_obj = sol['primal objective']
    
    # w calculated but not used
    w = ((np.multiply(RBF_kernel(X_train_sv,X_train_sv, gamma),np.multiply(alpha_star,y_train_sv))).sum(axis=0)).reshape((-1,1))
    
    # b
    b = np.mean((1-y_train_sv*sum(np.multiply(RBF_kernel(X_train_sv, X_train_sv, gamma),np.multiply(alpha_star,y_train_sv))))/y_train_sv)

    
    end_time = time.time()
    sec = end_time - start_time
    
    # Predict
    y_train_pred = np.sign(((np.multiply(RBF_kernel(X_train_sv, X_train, gamma),np.multiply(alpha_star,y_train_sv))).sum(axis=0)).reshape((-1,1))+(np.repeat(b, len(X_train), axis=0)).reshape((-1,1)))
    y_test_pred = np.sign(((np.multiply(RBF_kernel(X_train_sv, X_test, gamma),np.multiply(alpha_star,y_train_sv))).sum(axis=0)).reshape((-1,1))+(np.repeat(b, len(X_test), axis=0)).reshape((-1,1)))
    
    
    return C, gamma, alpha, funct_eva, final_obj, y_train_pred, y_test_pred, sec

####################################################################################################################################
def pred_error(y,y_pred):    
    conf_mat = confusion_matrix(y,y_pred, labels=[1, -1])
    cmtx = pd.DataFrame(conf_mat, index=['True label 2', 'True label 4'], columns=['Predicted label 2', 'Predicted lablel 4'])
    accuracy = sklearn.metrics.accuracy_score(y, y_pred)
    err = 1-accuracy
    return err, cmtx, accuracy
####################################################################################################################################
    
def CV(X,y,C,gamma,res):
    scores_tr = []
    scores_val = []
    
    
    cv = KFold(n_splits=10, random_state=1861402, shuffle=True)
    
    for train_index, val_index in cv.split(X):

        X_tr, X_val, y_tr, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
        #best_svr.fit(X_train, y_train)
        C, gamma, alpha, funct_eva, final_obj, y_tr_pred, y_val_pred, sec =  SVM(X_tr,y_tr,X_val, y_val, C, gamma)
        err_tr, conf_mat_tr, accuracy_tr = pred_error(y_tr,y_tr_pred)
        err_val, conf_mat_val, accuracy_val = pred_error(y_val,y_val_pred)
        scores_tr.append(err_tr)
        scores_val.append(err_val)
        
    return np.mean(scores_tr),np.mean(scores_val)
#####################################################################################################################################
   
def CV_exec(X_train,y_train):
# Hyper-parameters C, gamma
    C_values = [0.1,0.5,0.7,0.8,0.9,1.1,2.1]
    gamma_vals = [0.1,0.01,0.001,0.0001]
    
    all_poss_conf = []
    for c in C_values:
        for g in gamma_vals:
                all_poss_conf.append((c,g))
                
    print(all_poss_conf)
    
    res_df = pd.DataFrame (columns=['C', 'gamma', 'Training err', 'Validation err'])
    
    for k in all_poss_conf:
        err_tr, err_val = CV(X_train,y_train,k[0],k[1],res_df)
        res_df = res_df.append ({'C': k[0], 'gamma': k[1], 'Training err': err_tr,'Validation err': err_val}, ignore_index=True)
        
    print(res_df)
    res_df.to_csv('CV',header=True,index=False)
#################################################################################################################
def r_s(alpha_init,y,C,tollerance):
	lower_t= tollerance
	upper_t = C-tollerance# np.max(alpha_init)
	idx = np.where(alpha_init<=lower_t)[0]
	alpha_init[idx]=0
	idx = np.where(alpha_init>=upper_t)[0]
	alpha_init[idx]=C
	L_i = set(np.where(alpha_init==0)[0])
	U_i = set(np.where(alpha_init==C)[0])
	L_pos = L_i.intersection(set(np.where(y > 0)[0]))
	L_neg = L_i.intersection(set(np.where(y < 0)[0]))
	U_pos = U_i.intersection(set(np.where(y > 0)[0]))
	U_neg = U_i.intersection(set(np.where(y < 0)[0]))
	others = set(np.where(alpha_init<C)[0])
	others = others.intersection(set(np.where(alpha_init>0)[0]))
	R_alpha = (L_pos.union(U_neg)).union(others)
	S_alpha = (L_neg.union(U_pos)).union(others)
	return list(R_alpha),list(S_alpha)