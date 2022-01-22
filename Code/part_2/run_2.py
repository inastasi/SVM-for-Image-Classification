import os,sys,inspect
import Project_2_dataExtraction as pde
import numpy as np
import random
import sklearn
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from cvxopt import solvers
from cvxopt import matrix
import time
from sklearn.metrics import confusion_matrix
solvers.options['show_progress'] = False

np.random.seed(1861402)
random.seed(1861402)
xLabel2,yLabel2,xLabel4,yLabel4,xLabel6,yLabel6=pde.returnData()
yLabel2=np.array([1]*1000)
yLabel4=np.array([-1]*1000)
label2_data=np.append(xLabel2,yLabel2.reshape(1000,1),axis=1)
label4_data=np.append(xLabel4,yLabel4.reshape(1000,1),axis=1)
all_data=np.append(label2_data,label4_data,axis=0)
all_df = pd.DataFrame(all_data)
X = all_data[:,:-1]
Y = all_data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
gamma = 0.001
C = 2.1
P = X_train.shape[0]
def pred_error(y,y_pred):    
    conf_mat = confusion_matrix(y,y_pred)
    accuracy = sklearn.metrics.accuracy_score(y, y_pred)
    err = 1-accuracy
    return err, conf_mat, accuracy

def RBF_kernel(X,Y,gamma):
    return sk.metrics.pairwise.rbf_kernel(X, Y, gamma)

def r_s(alpha_init, y):
	lower_t= 0.00001
	upper_t = C-0.00001# np.max(alpha_init)
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

def SVM_init(X,y, C, gamma,q_size):
	alpha_init = np.zeros((P, 1))
	R, S = r_s(alpha_init, y_train)
	
	qR = random.choices(R, k=q_size//2)
	
	qS = random.choices(S, k=q_size//2)
	q = []
	q.extend(qR)
	q.extend(qS)
	q=list(set(q))
	idxx = np.array(range(1,P))
	
	idxx = set(idxx)
	W_hat_idx = list(idxx.difference(set(q)))
	
	alpha_w_hat = np.array(alpha_init[W_hat_idx]).reshape((len(W_hat_idx),1))
	y_hat = y_train[W_hat_idx,].reshape((-1,1))
	X_hat = X_train[W_hat_idx,]
	
	X= X_train[q,]
	
	y= y_train[q,].reshape((-1,1))
	
	K = RBF_kernel(X, X, gamma)
	np.reshape(y,(len(y),-1))
	Q1 = np.multiply(y,K)
	Q2 = np.multiply(y,Q1.T)
	Q = matrix(0.5*Q2.T)
	
	K = RBF_kernel(X_hat, X, gamma)
	np.reshape(y_hat,(len(y_hat),-1))
	Q1 = np.multiply(y_hat,K)
	Q2 = np.multiply(y,Q1.T)
	Q_hat = 0.5*Q2
	
	p = matrix(1*(Q_hat.dot(alpha_w_hat) - 1))
	
	A = matrix(y,(1,len(y)))
	# vector b is a scalar
	b = matrix(-y_hat.T.dot(alpha_w_hat),tc='d')

	# create the first constraint (- alpha <= 0)
	first_constr = np.diag([-1]*len(y))
	first_limit = np.array([0]*len(y))
	# create the second constraint (alpha <= C)
	second_constr = np.diag([1]*len(y))
	second_limit = np.array([C]*len(y))

	G = matrix(np.concatenate((first_constr, second_constr)),tc='d')
	h = matrix(np.concatenate((first_limit, second_limit)))

	# Solve minimization problem
	sol = solvers.qp(Q,p,G,h,A,b)

	# Take alpha from the solution
	alpha = np.array(sol['x'])
	alpha_init[q]=alpha
	# Support vectors are ones corresponding to alpha values greater than 1e-5
	ind = np.where(np.any(alpha>0.00001, axis=1))
	
	if len(list(ind))>1:
	    X_train_sv = X[ind]
	    y_train_sv = y[ind]
	    y_train_sv = (y_train_sv.T).reshape((-1,1))
	    alpha_star = alpha[ind]
	else:
		X_train_sv = X
		y_train_sv = y
		y_train_sv = (y_train_sv.T).reshape((-1,1))
		alpha_star = alpha
	
	funct_eva = sol["iterations"]
	final_obj = sol['primal objective']

	# KKT condition violation
	primal_inf = sol["primal infeasibility"]
	dual_inf = sol["dual infeasibility"]
	primal_slac = sol["primal slack"]
	dual_slac = sol["dual slack"]


	b = np.mean((1-y_train_sv*sum(np.multiply(RBF_kernel(X_train_sv, X_train_sv, gamma),np.multiply(alpha_star,y_train_sv))))/y_train_sv)

	y_train_pred = np.sign(((np.multiply(RBF_kernel(X_train, X_train, gamma),np.multiply(alpha_init,y_train.reshape((-1,1))))).sum(axis=0)).reshape((-1,1))+b )#(np.repeat(b, len(X_train), axis=0)).reshape((-1,1)))
	y_test_pred = np.sign(((np.multiply(RBF_kernel(X_train, X_test, gamma),np.multiply(alpha_init,y_train.reshape((-1,1))))).sum(axis=0)).reshape((-1,1))+b)#(np.repeat(b, len(X_test), axis=0)).reshape((-1,1)))
    # calc gradient
	alpha_grad = np.zeros((P,1))
	alpha_grad1 = (np.array(Q).dot(alpha) + np.array(Q_hat).dot(alpha_w_hat) - 1)
	alpha_grad2 = np.array(Q_hat).T.dot(alpha)
	alpha_grad[q]=alpha_grad1
	alpha_grad[W_hat_idx]=alpha_grad2
	
	return C, gamma, funct_eva, final_obj, y_train_pred, y_test_pred, primal_inf, dual_inf, primal_slac, dual_slac,alpha_init, alpha_grad


def SVM_iterations_02(X,y, C, gamma,q_size,prev_alpha, prev_alpha_grad):
	R,S = r_s(prev_alpha, y_train)
	grad_R = prev_alpha_grad[R].flatten()
	grad_S = prev_alpha_grad[S].flatten()
	fes=0
	for di in range(1,160):
		a= di * q_size
		sorting_S = np.argsort(grad_S)[a:a+(q_size)//2]
		sorting_R = np.argsort(-1*grad_R)[a:a+(q_size)//2]
		qS = [S[x] for x in sorting_S]
		qR = [R[x] for x in sorting_R]
		direction = np.zeros((P,1))
		direction[qR]=1/y_train[qR,].reshape((-1,1))
		direction[qS]=-1/y_train[qS,].reshape((-1,1))
		feasiblness= prev_alpha_grad.T.dot(direction)
		if feasiblness<0:
			fes = 1
			break
		
	if fes==0:
		prev_alpha_grad0 = -1*prev_alpha_grad/y_train.reshape((-1,1))
		grad_R = prev_alpha_grad0[R].flatten()
		grad_S = prev_alpha_grad0[S].flatten()
		sorting_S = np.argsort(grad_S)[:q_size//2]
		sorting_R = np.argsort(-1*grad_R)[:q_size//2]
		qS = [S[x] for x in sorting_S]
		qR = [R[x] for x in sorting_R]
	q = []
	q.extend(qR)
	q.extend(qS)
	q=list(set(q))[:q_size]
	idxx = np.array(range(1,P))
	idxx = set(idxx)
	W_hat_idx = list(idxx.difference(set(q)))
	alpha_init = prev_alpha
	alpha_w_hat = np.array(alpha_init[W_hat_idx]).reshape((len(W_hat_idx),1))
	y_hat = y_train[W_hat_idx,].reshape((-1,1))
	X_hat = X_train[W_hat_idx,]
	X= X_train[q,]
	y= y_train[q,].reshape((-1,1))
	K = RBF_kernel(X, X, gamma)
	np.reshape(y,(len(y),-1))
	Q1 = np.multiply(y,K)
	Q2 = np.multiply(y,Q1.T)
	Q = matrix(0.5*Q2.T)
	K = RBF_kernel(X_hat, X, gamma)
	np.reshape(y_hat,(len(y_hat),-1))
	Q1 = np.multiply(y_hat,K)
	Q2 = np.multiply(y,Q1.T)
	Q_hat = Q2
	p = matrix(1*(Q_hat.dot(alpha_w_hat) - 1))
	

	# vector A (1 x len(y_train))
	A = matrix(y,(1,len(y)))
	# vector b is a scalar
	b = matrix(-y_hat.T.dot(alpha_w_hat),tc='d')

	# create the first constraint (- alpha <= 0)
	first_constr = np.diag([-1]*len(y))
	first_limit = np.array([0]*len(y))
	# create the second constraint (alpha <= C)
	second_constr = np.diag([1]*len(y))
	second_limit = np.array([C]*len(y))

	G = matrix(np.concatenate((first_constr, second_constr)),tc='d')
	h = matrix(np.concatenate((first_limit, second_limit)))

	# Solve minimization problem
	sol = solvers.qp(Q,p,G,h,A,b)

	# Take alpha from the solution
	alpha = np.array(sol['x'])
	alpha_init[q]=alpha
	# Support vectors are ones corresponding to alpha values greater than 1e-5
	ind = np.where(np.any(alpha>0.00001, axis=1))
	if len(list(ind))>1:
	    X_train_sv = X[ind]
	    y_train_sv = y[ind]
	    y_train_sv = (y_train_sv.T).reshape((-1,1))
	    #y_train_sv= y_train_sv.reshape((y_train_sv.shape[0],1))    
	    alpha_star = alpha[ind]
	else:
		#print("else")
		X_train_sv = X
		y_train_sv = y
		y_train_sv = (y_train_sv.T).reshape((-1,1))
		alpha_star = alpha
	funct_eva = sol["iterations"]
	final_obj = sol['primal objective']

	# KKT condition violation
	primal_inf = sol["primal infeasibility"]
	dual_inf = sol["dual infeasibility"]
	primal_slac = sol["primal slack"]
	dual_slac = sol["dual slack"]


	b = np.mean((1-y_train_sv*sum(np.multiply(RBF_kernel(X_train_sv, X_train_sv, gamma),np.multiply(alpha_star,y_train_sv))))/y_train_sv)

	y_train_pred = np.sign(((np.multiply(RBF_kernel(X_train, X_train, gamma),np.multiply(alpha_init,y_train.reshape((-1,1))))).sum(axis=0)).reshape((-1,1))+b )
	y_test_pred = np.sign(((np.multiply(RBF_kernel(X_train, X_test, gamma),np.multiply(alpha_init,y_train.reshape((-1,1))))).sum(axis=0)).reshape((-1,1))+b)
	alpha_grad = np.zeros((P,1))
	alpha_grad1 = (np.array(Q).dot(alpha) + np.array(Q_hat).dot(alpha_w_hat) - 1 )
	alpha_grad2 = np.array(Q_hat).T.dot(alpha)
	alpha_grad[q]=alpha_grad1
	alpha_grad[W_hat_idx]=alpha_grad2 
	return C, gamma, funct_eva, final_obj, y_train_pred, y_test_pred, primal_inf, dual_inf, primal_slac, dual_slac,alpha_init, alpha_grad


def Q_proc(Q, C, gamma):
	start = time.time()	
	C, gamma, funct_eva0, final_obj, y_train_pred, y_test_pred, primal_inf, dual_inf, primal_slac, dual_slac,alpha, alpha_grad = SVM_init(X_train, y_train, C, gamma,Q)
	err_train, conf_mat_train, accuracy_train = pred_error(y_train, y_train_pred)
	err_test, conf_mat_test, accuracy_test = pred_error(y_test, y_test_pred)
	# for init val of obj func
	alpha_init = np.zeros((P, 1))
	Kernel_X = RBF_kernel(X_train,X_train,gamma)
	k2 = (Kernel_X*y_train)*y_train.T
	k1 =  0.5*k2*alpha_init * alpha_init.T
	init_val_obj = np.sum(k1)+np.sum(alpha_init)
	
	for i in range(1000):
		R,S = r_s(alpha, y_train)
		alpha_grad0 = -1*alpha_grad/y_train.reshape((-1,1))
		m1,m2 =(np.max(alpha_grad0[R]), np.min(alpha_grad0[S]))
		
		if (abs(m1-m2)>=0.00001):
			
			C, gamma, funct_eva, final_obj, y_train_pred, y_test_pred, primal_inf, dual_inf, primal_slac, dual_slac,alpha, alpha_grad = SVM_iterations_02(X_train, y_train, C, gamma,Q, alpha, alpha_grad)
			funct_eva0 = funct_eva0+funct_eva
		else:
			break
			
	err_train, conf_mat_train, accuracy_train = pred_error(y_train, y_train_pred)
	err_test, conf_mat_test, accuracy_test = pred_error(y_test, y_test_pred)
	print("q value: %s"%(Q))
	print("Gamma value: %s"%(gamma))
	print("C value: %s"%(C))
	print("Accuracy Rate Training Set: %s"%(accuracy_train))
#	print("Training Error: ",err_train)
#	print("Confusion matrix: \n", conf_mat_train)
	print("-------------------------------")
	print("Accuracy Rate Test Set: %s"%(accuracy_test))
#	print("Test Error: ",err_test)
	print("Confusion matrix: \n", conf_mat_test)
	print("-------------------------------")
	print("Time: %s seconds"%(time.time()-start))
	print("Number of iterations: ",i)
	#print("Number of function evaluations: ", funct_eva0)
	#print("initial obj fun value", init_val_obj)
	#print("Final value of the objective function of the dual problem: ", final_obj )
	#print("Primal infieasibility: ", primal_inf)
	#print("Dual infieasibility: ", dual_inf)
	#print("Primal slac: ", primal_slac)
	#print("Dual slac: ", dual_slac)
	print("m-M: ", m1-m2)
#    print("init_val_obj: ", init_val_obj)
#	print("final obj_fun: ", final_obj)
 

print(Q_proc(10, C, gamma))

