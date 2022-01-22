# -*- coding: utf-8 -*-
import sklearn as sk
import numpy as np
import time
import sklearn
from sklearn.metrics import confusion_matrix
import pandas as pd

def RBF_kernel(X,Y,gamma):
    return sk.metrics.pairwise.rbf_kernel(X, Y, gamma)

##############################################################################################################

def Q_mat(X,Y,gamma):
    K = RBF_kernel(X, X, gamma)
    np.reshape(Y,(len(Y),-1))
    Q1 = np.multiply(Y,K)
    Q = np.multiply(Y,Q1.T).T
    #Q = matrix(Q2.T)
    return Q

##############################################################################################################

def grad(Q,alpha):
    return np.dot(Q,alpha)-1

##############################################################################################################

def pred_error(y,y_pred):    
    conf_mat = confusion_matrix(y,y_pred, labels=[1, -1])
    cmtx = pd.DataFrame(conf_mat, index=['True label 2', 'True label 4'], columns=['Predicted label 2', 'Predicted lablel 4'])
    accuracy = sklearn.metrics.accuracy_score(y, y_pred)
    err = 1-accuracy
    return err, cmtx, accuracy

##############################################################################################################

def prediction_accuracy(X,Y,test,test_y, alpha,gamma,eps=0.00001):
    ind = np.where(np.any(alpha>eps, axis=1))
    
    X_sv = X[ind]
    Y_sv = (Y[ind].T).reshape((-1,1))
    
    alpha_star = alpha[ind]
    
    b = np.mean((1-Y_sv*sum(np.multiply(RBF_kernel(X_sv, X_sv, gamma),np.multiply(alpha_star,Y_sv))))/Y_sv)

    
    # Predict
    Y_pred = np.sign(((np.multiply(RBF_kernel(X_sv, X, gamma),np.multiply(alpha_star,Y_sv))).sum(axis=0)).reshape((-1,1))+(np.repeat(b, len(X), axis=0)).reshape((-1,1)))
    test_pred = np.sign(((np.multiply(RBF_kernel(X_sv, test, gamma),np.multiply(alpha_star,Y_sv))).sum(axis=0)).reshape((-1,1))+(np.repeat(b, len(test), axis=0)).reshape((-1,1)))
    
    err_tr, conf_mat_tr, accuracy_tr = pred_error(Y,Y_pred)
    err_te, conf_mat_te, accuracy_te = pred_error(test_y,test_pred)
    
    return err_tr, conf_mat_tr, accuracy_tr, err_te, conf_mat_te, accuracy_te

##############################################################################################################

def r_s(alpha_init, y, C, tollerance):
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

##############################################################################################################
def print_res(gamma, C, q, accuracy_train, conf_mat_train, accuracy_test, conf_mat_test, sec, itr,m,M):
    print("Gamma value: %s"%(gamma))
    print("C value: %s"%(C))
    print("q value: %s"%(q))
    print("-------------------------------")
    print("Accuracy Rate Training Set: %s"%(accuracy_train))
    #print("Training Error: ",err_train)
    print("Confusion matrix: \n", conf_mat_train)
    print("-------------------------------")
    print("Accuracy Rate Test Set: %s"%(accuracy_test))
    #print("Test Error: ",err_test)
    print("Confusion matrix: \n", conf_mat_test)
    print("-------------------------------")
    print("Time: %s seconds"%(sec))
    print("Number of iterations: %s"%(itr))
    print("KKT violation: %s"%(m-M))
    
##############################################################################################################
def MVP(X_train,y_train,X_test,y_test, gamma, q, tollerance, C, maxiter=10000):
    start_time = time.time()
    P=X_train.shape[0]
        
    #initialize alpha
    alpha_k = np.zeros((P,1))
    
    Q = Q_mat(X_train,y_train,gamma)
    
    k=0
    opt=False
    
    while(opt!=True):
        #######################################################
        #Create working set w
        #######################################################
        #choose/calculate R and S
        R,S = r_s(alpha_k,y_train,C,tollerance)
        
        
        grad_k = grad(Q,alpha_k)
        
        grad_y = -np.multiply(grad_k,y_train)
        
        m = max(np.take(grad_y, R))
        M = min(np.take(grad_y, S))
           
        # Constructing working set
        m_index = np.where(grad_y==m)[0][0] #one index where gradient is equal to m
        M_index = np.where(grad_y==M)[0][0] #one index where gradient is equal to M
        w_k = [m_index, M_index]
        
        ####################################################
        #Reduce problem dimansion to working set
        ####################################################
        y_train_k = y_train[w_k]
        X_train_k = X_train[w_k]
        Q_k = Q[w_k,:][:,w_k]
        alpha_k_n=alpha_k[w_k]
        
        
        dij=np.array([y_train_k[0],-y_train_k[1]])
                
        a1 = alpha_k_n[0,0]
        a2 = alpha_k_n[1,0]
        d1 = dij[0,0]
        d2 = dij[1,0]
        
        ####################################################
        #t feasible
        ####################################################
        if d1 > 0:
            if d2 > 0:
                t = min(C-a1, C-a2)
            else:
                t = min(C-a1, a2)
        else:
            if d2 > 0:
                t = min(a1, C-a2)
            else:
                t = min(a1, a2)
                
        ####################################################
        #t star
        ####################################################
        if np.dot(grad_k[w_k].T,dij) == 0:
            t_star=0
        else:
            if np.dot(grad_k[w_k].T,dij) < 0:
                d_star=dij
            else:
                d_star=-dij
            
            if t == 0:
                t_star = 0
            elif np.dot(np.dot(d_star.T,Q_k),d_star) == 0:
                t_star = t
            else:
                if np.dot(np.dot(d_star.T,Q_k),d_star) > 0:
                    t_max = ((np.dot(-grad_k[w_k].T,d_star))/(np.dot(np.dot(d_star.T,Q_k),d_star)))[0,0]
                    t_star= min(t,t_max)
        
        # Move alpha on coordinates from working set w
        alpha_star=alpha_k_n+np.dot(t_star,d_star)
        
        # Update alpha   
        alpha_k[w_k]=alpha_star
        
        grad_n = grad(Q,alpha_k)
        
        R,S = r_s(alpha_k,y_train,C,tollerance)
        
        grad_y = -np.multiply(grad_n,y_train)
        
        m = max(np.take(grad_y, R))
        M = min(np.take(grad_y, S))
              
        
        k += 1
        
        # Check first stopping criteria
        if m - M <= tollerance:
            opt = True
            end_time = time.time()
            sec=end_time-start_time
            err_tr, conf_mat_tr, accuracy_tr, err_te, conf_mat_te, accuracy_te = prediction_accuracy(X_train,y_train,X_test,y_test, alpha_k , gamma,tollerance)
            final_obj = np.dot(np.dot(alpha_k.T,Q),alpha_k)*0.5-np.sum(alpha_k)
            print_res(gamma, C, q, accuracy_tr, conf_mat_tr, accuracy_te, conf_mat_te, sec, k,m,M)
            return final_obj
        if k == maxiter:
            opt = True
            end_time = time.time()
            sec=end_time-start_time
            err_tr, conf_mat_tr, accuracy_tr, err_te, conf_mat_te, accuracy_te = prediction_accuracy(X_train,y_train,X_test,y_test, alpha_k , gamma,tollerance)
            final_obj = np.dot(np.dot(alpha_k.T,Q),alpha_k)*0.5-np.sum(alpha_k)
            print_res(gamma, C, q, accuracy_tr, conf_mat_tr, accuracy_te, conf_mat_te, sec, k,m,M)
            return final_obj

