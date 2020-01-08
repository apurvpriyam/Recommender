import numpy as np
import random
import time
import math

def rmse(u, v, mat):
    mask = mat > 0
    res = np.sum(((u.dot(v.T) - mat) * mask) ** 2) / float(np.sum(mask))
    return np.sqrt(res)

def my_recommender(rate_mat, lr, with_reg):
    """

    :param rate_mat:
    :param lr:
    :param with_reg:
        boolean flag, set true for using regularization and false otherwise
    :return:
    """

    # new parameters defined
    run_time = 3*60     # Total time to be used for training (in seconds)
    start = time.time() # Time at which the training was started
    
    # TODO pick hyperparams
    max_iter = 150
    learning_rate = 0.001
    reg_coef =  0.00004
    
    if not with_reg:
        reg_coef = 0
    n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]
 
    np.random.seed(0)
    U = np.random.rand(n_user, lr) / lr
    V = np.random.rand(n_item, lr) / lr
 
    # TODO implement your code here ----------------------------
    # creating training  and CV data for hyper-parameter tuning: =============

    # ~~~~~~ UNCOMMENT FOR CREATING CV DATA FOR HYPERPARAMETER TUNING ~~~~~~~~
    # ~~~~~~ Minor modifications at other places may be required ~~~~~~~~~~~~~
    
    # rate_mat = pd.DataFrame(rate_mat)
    # rate_mat['user'] = range(n_user)
    # rate_mat = rate_mat.melt(id_vars=['user'], var_name='item', value_name='rating')
    # rate_mat_f = rate_mat.loc[rate_mat['rating']>0]
    
    # msk = np.random.rand(len(rate_mat_f)) < 1.1 # while hyperparameter tuning this was used to filter 80% data
    # train = rate_mat_f[msk]
    # train = train.reset_index()
    # cv = rate_mat_f                             #[~msk] while hyperparameter tuning this was used to filter 20% data
    # cv = cv.reset_index()
    
    # # bringing all combinations of user/item
    # train_mat = train.merge(rate_mat[rate_mat.columns[0:2]], on = ['user', 'item'], how = 'outer')
    # train_mat.fillna(0, inplace=True)
    # # changing the shape to n_user, n_item
    # train_mat = pd.pivot_table(train_mat, values = 'rating', index=['user'], columns = 'item',fill_value=0).reset_index()
    # train_mat = train_mat.drop(['user'], axis=1)

    # rate_mat = train_mat

    # =========================================================================
    
    train = rate_mat
    # Filtering for only those user-item for which we have rating.
    # We don't want to train from the non-rated pair
    samples_full = []
    for i in range(n_user):
        for j in range(n_item):
            if (train[i,j] > 0):
                samples_full.append((i,j,train[i,j]))
 
    rmse_prev = 100
    
    for zz in range(max_iter):

        lamb = math.exp(-(0.6*(zz+1))) # decrasing learning rate; started from higher value
        if (lamb > 0.01): # putting upper bound on learning rate of 0.01
            lamb = 0.01
        if (lamb < learning_rate):    # lower bound of 0.001
            lamb = learning_rate
        
        random.shuffle(samples_full)
        for user, item, rate in samples_full:
            delta_m = rate - U[user, :].dot(V[item, :].T)
            U[user,:] = U[user,:] + 2*lamb*delta_m*V[item,:] - 2*reg_coef*U[user,:]
            V[item,:] = V[item,:] + 2*lamb*delta_m*U[user,:] - 2*reg_coef*V[item,:]

        # =============================================================================== 
        # ~~~~~~ UNCOMMENT FOR CREATING CV DATA FOR HYPERPARAMETER TUNING ~~~~~~~~
        # ~~~~~~ Minor modifications at other places may be required ~~~~~~~~~~~~~
        # # creating CV subset by randomly select 80% data from CV data 
        # # to prevent data leak from CV to training data

        # # msk = np.random.rand(len(cv)) < 0.8 # while hyperparameter tuning this was used
        # CV1 = cv  #[msk] # for final submission complete train data is used to calculate rmse
  
        # # bringing all combinations of user/item
        # CV1 = CV1.merge(rate_mat[rate_mat.columns[0:2]], on = ['user', 'item'], how = 'outer')
        # CV1.fillna(0, inplace=True)
        # # changin shape to n_user, n_item
        # CV1 = pd.pivot_table(CV1, values = 'rating', index=['user'], columns = 'item',fill_value=0).reset_index()
        # CV1 = CV1.drop(['user'], axis=1)
        # ===============================================================================
        
        # If 3 minutes ia almost over, stop the training
        if (time.time() - start > run_time-10):
            break
        # Improvement in RMSE check
        rmse_current = rmse(U,V,train)
        if (rmse_prev - rmse_current)/rmse_prev < 0.0001:
            break
        else:
            U_out = U
            V_out = V
            rmse_prev = rmse_current
        
    return U_out, V_out