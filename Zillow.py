
# coding: utf-8

# # Zillow House Price Prediction Model
# 

# ## Packages

# In[1]:

import numpy as np
from scipy import sparse,stats
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, pipeline, metrics
import seaborn as sns
import time
import os

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ## Loading Data

# In[3]:

properties = pd.read_csv(r'./Data/properties_2016.csv')
train = pd.read_csv(r"./Data/train_2016_v2.csv")


# In[14]:

properties.columns, train.columns


# In[15]:

train.shape, properties.shape


# In[16]:

train_df = train.merge(properties, how='left', on='parcelid')


# ## Exploratory Data Analysis

# ### logerror

# In[17]:

plt.figure(figsize=(10,5))
sns.distplot(train_df.logerror, bins=50, kde=False)
plt.xlabel('logerror', fontsize=10)


# In[18]:

ulimit = np.percentile(train_df.logerror.values, 99)
llimit = np.percentile(train_df.logerror.values, 1)

plt.figure(figsize=(10,5))
sns.distplot(train_df.query('logerror<%f and logerror>%f' % (ulimit, llimit)).logerror.values, bins=50, kde=False)
plt.xlabel('logerror', fontsize=8)


# ### taxvaluedollarcnt

# In[19]:

plt.figure(figsize=(10,5))
sns.distplot(train_df.query('taxvaluedollarcnt==taxvaluedollarcnt').taxvaluedollarcnt, bins=50, kde=False)
plt.xlabel('taxvaluedollarcnt', fontsize=10)


# In[20]:

ulimit = np.percentile(train_df.query('taxvaluedollarcnt==taxvaluedollarcnt').taxvaluedollarcnt.values, 99)
llimit = np.percentile(train_df.query('taxvaluedollarcnt==taxvaluedollarcnt').taxvaluedollarcnt.values, 1)


plt.figure(figsize=(10,5))
sns.distplot(train_df.query('taxvaluedollarcnt<%f and taxvaluedollarcnt>%f' % (ulimit, llimit)).taxvaluedollarcnt.values, bins=50, kde=False)
plt.xlabel('taxvaluedollarcnt', fontsize=10)


# ### lotsize

# In[21]:

plt.figure(figsize=(10,5))
sns.distplot(train_df.query('lotsizesquarefeet==lotsizesquarefeet and lotsizesquarefeet>0 and lotsizesquarefeet<50000').lotsizesquarefeet.values, bins=50, kde=False)
plt.xlabel('lotsizesquarefeet', fontsize=12)
plt.show()


# ### built year

# In[22]:

plt.figure(figsize=(10,5))
sns.distplot(train_df.query('yearbuilt==yearbuilt ').yearbuilt, bins=50, kde=False)
plt.xlabel('yearbuilt', fontsize=10)
plt.show()


# ### Numerical vs numerical : log error vs. tax value

# In[25]:

sns.jointplot('taxvaluedollarcnt','logerror',
              train_df.query('taxvaluedollarcnt==taxvaluedollarcnt'),size=6,kind='reg')


# In[26]:

train_df['abs_log_error'] = train_df.logerror.abs()


# In[27]:

sns.jointplot('taxvaluedollarcnt','abs_log_error',
              train_df.query('taxvaluedollarcnt==taxvaluedollarcnt'),size=6,kind='reg')


# ### log error vs built year

# In[29]:

sns.jointplot('yearbuilt','abs_log_error',train_df.groupby('yearbuilt').abs_log_error.mean().reset_index(),size=6,kind='reg')


# ###  numerical vs categorical

# In[30]:

plt.figure(figsize=(20, 10))
sns.violinplot(data=train_df,
            x='architecturalstyletypeid',
            y='abs_log_error')


# #### Architectual style type id 7: remove outliers

# ### Missing values:

# In[55]:

missing_df = properties.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(6,9))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# ## Data Preprocessing

# ### Group Variables

# #### Separate categorical, numerical, id, and target features. 

# In[32]:

data_types = train_df.dtypes  
cat_vars = list(data_types[data_types=='object'].index)
num_vars = list(data_types[data_types=='int64'].index) + list(data_types[data_types=='float64'].index)

id_var = 'id'
target_var = 'logerror'
num_vars.remove('parcelid')
num_vars.remove('logerror')
num_vars.remove('abs_log_error')

## transactiondate is actually a datetime feature
cat_vars.remove('transactiondate')

dt_vars=[]
dt_vars.append('transactiondate')


print ("Categorical features:", cat_vars)
print ( "Numerical features:", num_vars)
print ( "Datetime features:", dt_vars)
print ( "ID: %s, target: %s" %( id_var, target_var))


# ### Feature Interactions - numerical to numerical

# In[33]:

properties['finished_sq_ratio'] = properties[['calculatedfinishedsquarefeet','lotsizesquarefeet']].apply(lambda x:x[0]/x[1] if x[1]>0 else -999999,axis=1)

properties['taxvalue_per_sq'] = properties[['taxvaluedollarcnt','calculatedfinishedsquarefeet']].apply(lambda x:x[0]/x[1] if x[1]>0 else -999999,axis=1)

properties['structure_tax_ratio'] = properties[['structuretaxvaluedollarcnt','taxvaluedollarcnt']].apply(lambda x:x[0]/x[1] if x[1]>0 else -999999,axis=1)

properties['landtax_per_sq'] = properties[['landtaxvaluedollarcnt','lotsizesquarefeet']].apply(lambda x:x[0]/x[1] if x[1]>0 else -999999,axis=1)

properties['assessmentyear_to_builtyear']= properties[['assessmentyear','yearbuilt']].apply(lambda x:x[0]-x[1] if x[0]>0 and x[1]>0 else -999999,axis=1)


num_to_num_vars = ['finished_sq_ratio','taxvalue_per_sq','structure_tax_ratio',
                   'landtax_per_sq','assessmentyear_to_builtyear']


# In[34]:

train_df = pd.merge(train_df, properties[num_to_num_vars + ['parcelid']],
                     how='left', on='parcelid')


# In[ ]:




# ### Categorical Features: Label Encoding

# In[35]:

LBL = preprocessing.LabelEncoder()

LE_vars=[]
LE_map=dict()
for cat_var in cat_vars:
    print ("Label Encoding %s" % (cat_var))
    LE_var=cat_var+'_le'
    properties[LE_var]=LBL.fit_transform(properties[cat_var].astype(str).fillna('none'))
    LE_vars.append(LE_var)
    LE_map[cat_var]=LBL.classes_
    
print ("Label-encoded feaures: %s" % (LE_vars))


# ### Categorical Features: One Hot Encoding

# In[36]:

OHE = preprocessing.OneHotEncoder(sparse=True)
start=time.time()
OHE.fit(properties[LE_vars])
OHE_sparse=OHE.transform(properties[LE_vars])
                                   
print ('One-hot-encoding finished in %f seconds' % (time.time()-start))


OHE_vars = [var[:-3] + '_' + str(level).replace(' ','_')                for var in cat_vars for level in LE_map[var] ]

print ("OHE_sparse size :" ,OHE_sparse.shape)
print ("One-hot encoded catgorical feature samples : %s" % (OHE_vars[:100]))


# In[ ]:




# ## Modelling

# ### Cross-validation

# In[37]:

train_df = train.merge(properties, how='left', on='parcelid')


# In[38]:

full_vars = num_vars + LE_vars 
train_x=train_df[full_vars]
train_y = train_df['logerror'].values.astype(np.float32)

test_x = properties[full_vars]

# xgboost params
xgb_params = {
    'eta': 0.05,
    'max_depth': 4,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1,
    'seed': 1234
}

dtrain = xgb.DMatrix(train_x, train_y)
dtest = xgb.DMatrix(test_x)

# cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   nfold=5,
                   num_boost_round=10000,
                   early_stopping_rounds=50,
                   verbose_eval=10, 
                   show_stdv=False,
                   seed = 1234
                  )

## best score and best round
best_iteration = len(cv_result)
best_score = cv_result['test-mae-mean'].min()
print("Best score %f, best iteration %d" % (best_score,best_iteration) )


# ### Training Baseline Model:

# In[50]:

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=best_iteration)
pred = model.predict(dtest)
y_pred=[]

for i,predict in enumerate(pred):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
from datetime import datetime
output.to_csv('./output/sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print ("Finished")


# In[61]:

feature_imporantce = pd.Series(model.get_fscore()).sort_values(ascending=True)
#simply sums up how many times each feature is split on
feature_imporantce.plot.barh(x='feature_name',figsize=(10,15))


# In[ ]:




# #### Submitted Score of Baseline Model: *0.0649385*, Leader Board Ranking: 60%

# ### Additional Feature Engineering

# #### Imputation of Missing Values in Built Year.

# In[41]:

from sklearn.linear_model import Ridge,ElasticNet, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
def search_model(train_x, train_y, est, param_grid, n_jobs, cv, refit=False):
##Grid Search for the best model
    model = GridSearchCV(estimator  = est,
                                     param_grid = param_grid,
                                     scoring    = 'neg_mean_absolute_error',
                                     verbose    = 10,
                                     n_jobs  = n_jobs,
                                     iid        = True,
                                     refit    = refit,
                                     cv      = cv)
    # Fit Grid Search Model
    model.fit(train_x, train_y)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:", model.best_params_)
    print("Scores:", model.grid_scores_)
    return model


# In[42]:

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor


# In[43]:

train_build_year = properties.query('yearbuilt==yearbuilt')[['parcelid','longitude','latitude', 'yearbuilt']]
test_build_year = properties.query('yearbuilt!=yearbuilt and longitude==longitude and latitude==latitude')[['parcelid','longitude','latitude']]


## Simple parameter tuning by grid search
param_grid = {
              "n_neighbors":[1,2,3,4,5]
              }
model = search_model(train_build_year[['longitude','latitude']].values
                                         , train_build_year['yearbuilt'].values
                                         , KNeighborsRegressor()
                                         , param_grid
                                         , n_jobs=-1  # Cancel paralle processing. 
                                         , cv=5
                                         , refit=True)   

print ("best subsample:", model.best_params_)


pred = model.predict(test_build_year[['longitude','latitude']].values)
test_build_year['yearbuilt_pred'] = pred

properties = pd.merge(properties, test_build_year[['parcelid','yearbuilt_pred']], how='left', on='parcelid')

properties['yearbuilt'] = properties[['yearbuilt','yearbuilt_pred']].apply(lambda x: x[1] if x[0]!=x[0] else x[0], axis=1)


# ### Automatic Model Tuning using Bayesian Optimization:

# In[45]:

full_vars = num_vars + LE_vars
train_df = pd.merge(train, properties,
                     how='left', on='parcelid')

#Exclude outliers
train_df=train_df[ train_df.logerror > -0.4 ]
train_df=train_df[ train_df.logerror < 0.418 ]

xgtrain = xgb.DMatrix(train_df[full_vars].values, 
                         train_df['logerror'].values)
xgtest = xgb.DMatrix(properties[full_vars].values)

train_x = train_df[full_vars].values
train_y = train_df['logerror'].values

test_x = properties[full_vars].values


# ### XGBoost Tuning

# In[ ]:

from bayes_opt import BayesianOptimization
def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma):
    params = dict()
    params['objective'] = 'reg:linear'
    params['eta'] = 0.1
    params['max_depth'] = int(max_depth )   
    params['min_child_weight'] = int(min_child_weight)
    params['colsample_bytree'] = colsample_bytree
    params['subsample'] = subsample
    params['gamma'] = gamma
    params['verbose_eval'] = True    


    cv_result = xgb.cv(params, xgtrain,
                       num_boost_round=100000,
                       nfold=5,
                       metrics={'mae'},
                       seed=1234,
                       callbacks=[xgb.callback.early_stop(50)])

    return -cv_result['test-mae-mean'].min()


xgb_BO = BayesianOptimization(xgb_evaluate, 
                             {'max_depth': (2, 5),
                              'min_child_weight': (0, 100),
                              'colsample_bytree': (0.1, 1),
                              'subsample': (0.7, 1),
                              'gamma': (0, 2)
                             }
                            )

xgb_BO.maximize(init_points=8, n_iter=40)


# In[ ]:

BO_scores = pd.DataFrame(xgb_BO.res['all']['params'])
BO_scores['score'] = pd.DataFrame(xgb_BO.res['all']['values'])
BO_scores = BO_scores.sort_values(by='score',ascending=False).reset_index()
BO_scores.head()


# ### LightGBM Tuning

# In[47]:

import lightgbm as lgb


# In[ ]:

lgb_train = lgb.Dataset(train_df[full_vars].values, 
                         train_df['logerror'].values)

def lgb_evaluate(max_bin,
                 num_leaves,
                 min_sum_hessian_in_leaf,
                 min_gain_to_split,
                 feature_fraction,
                 bagging_fraction,
                 bagging_freq
                 ):
    params = dict()
    params['objective'] = 'regression'
    params['learning_rate'] = 0.01
    params['max_bin'] = int(max_bin)   
    params['num_leaves'] = int(num_leaves)    
    params['min_sum_hessian_in_leaf'] = int(min_sum_hessian_in_leaf)
    params['min_gain_to_split'] = min_gain_to_split    
    params['feature_fraction'] = feature_fraction
    params['bagging_fraction'] = bagging_fraction
    params['bagging_freq'] = int(bagging_freq)


    cv_results = lgb.cv(params,
                    lgb_train,
                    num_boost_round=100000,
                    nfold=5,
                    early_stopping_rounds=100,
                    metrics='mae',
                    shuffle=False,
                    verbose_eval=False
                   )
    return -pd.DataFrame(cv_results)['l1-mean'].min()


lgb_BO = BayesianOptimization(lgb_evaluate, 
                             {'max_bin': (127, 1023),
                              'num_leaves': (15, 512),
                              'min_sum_hessian_in_leaf': (1, 100),
                              'min_gain_to_split': (0,2),
                              'feature_fraction': (0.2, 0.8),
                              'bagging_fraction': (0.7, 1),
                              'bagging_freq': (1, 5)
                             }
                            )

lgb_BO.maximize(init_points=5, n_iter=40)


# In[ ]:

lgb_BO_scores = pd.DataFrame(lgb_BO.res['all']['params'])
lgb_BO_scores['score'] = pd.DataFrame(lgb_BO.res['all']['values'])
lgb_BO_scores = lgb_BO_scores.sort_values(by='score',ascending=False).reset_index()
lgb_BO_scores.head()


# ## Model Ensembling using Top 3 XGBoost and LightGBM Models.

# ### Top 3 XGBoost Models

# In[ ]:

xgb_params = []
for i in range(3):
    params = dict()
    params['max_depth'] = int(BO_scores['max_depth'][i])
    params['min_child_weight'] = int(BO_scores['min_child_weight'][i])
    params['colsample_bytree'] = BO_scores['colsample_bytree'][i]
    params['subsample'] = BO_scores['subsample'][i]
    params['gamma'] = BO_scores['gamma'][i]

    params['objective'] = 'reg:linear'
    params['eta'] = 0.01
    params['num_boost_round'] = 1200
    params['seed'] = 1234
    xgb_params.append(params)
print (xgb_params)


# ### Top 3 LightGBM models

# In[ ]:

lgb_params = []
for i in range(3):
    params=dict()
    params['max_bin'] = int(lgb_BO_scores['max_bin'][i])
    params['num_leaves'] = int(lgb_BO_scores['num_leaves'][i])
    params['min_sum_hessian_in_leaf'] = int(lgb_BO_scores['min_sum_hessian_in_leaf'][i])
    params['min_gain_to_split'] = lgb_BO_scores['min_gain_to_split'][i]
    params['feature_fraction'] = lgb_BO_scores['feature_fraction'][i]
    params['bagging_fraction'] = lgb_BO_scores['bagging_fraction'][i]
    params['bagging_freq'] = int(lgb_BO_scores['bagging_freq'][i])

    params['objective'] = 'regression'
    params['learning_rate'] = 0.01
    params['num_boost_round']=80
    params['seed'] = 1234
    lgb_params.append(params)
print (lgb_params)


# ### Model Stacking

# In[48]:

from sklearn.model_selection import StratifiedKFold, KFold 
def xgb_rgr_stack(rgr_params, train_x, train_y, test_x, kfolds, early_stopping_rounds=0, missing=None):

    skf = KFold(n_splits=kfolds,random_state=1234)
    skf_ids = list(skf.split(train_y))


    train_blend_x = np.zeros((train_x.shape[0], len(rgr_params)))
    test_blend_x = np.zeros((test_x.shape[0], len(rgr_params)))
    blend_scores = np.zeros ((kfolds,len(rgr_params)))

    print  ("Start stacking.")
    for j, params in enumerate(rgr_params):
        print ("Stacking model",j+1, params)
        test_blend_x_j = np.zeros((test_x.shape[0]))
        for i, (train_ids, val_ids) in enumerate(skf_ids):
            start = time.time()
            print ("Model %d fold %d" %(j+1,i+1))
            train_x_fold = train_x[train_ids]
            train_y_fold = train_y[train_ids]
            val_x_fold = train_x[val_ids]
            val_y_fold = train_y[val_ids]
            
            
            if early_stopping_rounds==0:
                model = xgb.train(params,
                                    xgb.DMatrix(train_x_fold, 
                                                label=train_y_fold.reshape(train_y_fold.shape[0],1), 
                                                missing=missing),
                                    num_boost_round=params['num_boost_round']
                                )
                val_y_predict_fold = model.predict(xgb.DMatrix(val_x_fold,missing=missing))

                score = metrics.mean_absolute_error(val_y_fold,val_y_predict_fold)
                print ("Score: ", score)
                blend_scores[i,j]=score
                train_blend_x[val_ids, j] = val_y_predict_fold
                test_blend_x_j = test_blend_x_j + model.predict(xgb.DMatrix(test_x,missing=missing))
                print (time.time()-start)
        test_blend_x[:,j] = test_blend_x_j/kfolds
        print ("Score for model %d is %f" % (j+1,np.mean(blend_scores[:,j])))
    return train_blend_x, test_blend_x, blend_scores    


def lgb_rgr_stack(rgr_params, train_x, train_y, test_x, kfolds, early_stopping_rounds=0, missing=None):

    skf = KFold(n_splits=kfolds,random_state=1234)
    skf_ids = list(skf.split(train_y))


    train_blend_x = np.zeros((train_x.shape[0], len(rgr_params)))
    test_blend_x = np.zeros((test_x.shape[0], len(rgr_params)))
    blend_scores = np.zeros ((kfolds,len(rgr_params)))

    print  ("Start stacking.")
    for j, params in enumerate(rgr_params):
        print ("Stacking model",j+1, params)
        test_blend_x_j = np.zeros((test_x.shape[0]))
        for i, (train_ids, val_ids) in enumerate(skf_ids):
            start = time.time()
            print ("Model %d fold %d" %(j+1,i+1))
            train_x_fold = train_x[train_ids]
            train_y_fold = train_y[train_ids]
            val_x_fold = train_x[val_ids]
            val_y_fold = train_y[val_ids]
            
            
            if early_stopping_rounds==0:
                model = lgb.train(params,
                                    lgb.Dataset(train_x_fold, 
                                                 train_y_fold),
                                    num_boost_round=params['num_boost_round']
                                )
                val_y_predict_fold = model.predict(val_x_fold)

                score = metrics.mean_absolute_error(val_y_fold,val_y_predict_fold)
                print ("Score for Model %d fold %d: %f " % (j+1,i+1,score))
                blend_scores[i,j]=score
                train_blend_x[val_ids, j] = val_y_predict_fold
                test_blend_x_j = test_blend_x_j + model.predict(test_x)
                print ("Model %d fold %d finished in %d seconds." % (j+1,i+1, time.time()-start))
        test_blend_x[:,j] = test_blend_x_j/kfolds
        print ("Score for model %d is %f" % (j+1,np.mean(blend_scores[:,j])))
    return train_blend_x, test_blend_x, blend_scores


# ### Level 1 Stacking

# In[ ]:

train_blend_x_xgb, test_blend_x_xgb, blend_scores_xgb = xgb_rgr_stack(xgb_params, train_x, train_y, test_x, 5, early_stopping_rounds=0, missing=None)
train_blend_x_lgb, test_blend_x_lgb, blend_scores_lgb = lgb_rgr_stack(lgb_params, train_x, train_y, test_x, 5, early_stopping_rounds=0, missing=None)


# ### Level 2 Stacking

# In[ ]:

param_grid = {
              "alpha":[0.001,0.01,0.1,1,10,30,100]
              }
model = search_model(train_blend_x
                                         , train_y
                                         , Ridge()
                                         , param_grid
                                         , n_jobs=-1
                                         , cv=5
                                         , refit=True)   

print ("best subsample:", model.best_params_)


pred = model.predict(test_blend_x)

y_pred=[]

## pred_ridge_l2 will be used later
pred_ridge_l2 = pred

for i,predict in enumerate(pred):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
from datetime import datetime
output.to_csv('../output/sub_lgb_xgb_1_ridge_l2_LE{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)

print ("Finished")


# #### Submitted Score of Ensemble Model: *0.0645138*, Leader Board Ranking: 20%

# In[ ]:




# In[ ]:



