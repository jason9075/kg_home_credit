#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:06:25 2018

@author: jason9075
"""
# https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from util import missing_values_table
from util import kde_target
from util import one_hot_encoder
from util import useless_columns

# sklearn preprocessing for dealing with categorical variables
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from xgboost import XGBClassifier

data_train = pd.read_csv('application_train.csv') #shape=(307511, 122)
# T=0: 282686, T=1:24825
data_test = pd.read_csv('application_test.csv') #shape=(48744, 121)

#info

# Missing values statistics
missing_values_table(data_train)
missing_values_table(data_test)


# Number of each type of column
data_train.dtypes.value_counts() # float=65, int=41, object=16

# Number of unique classes in each object column
data_train.select_dtypes(include=['object']).apply(pd.Series.nunique, axis = 0)


#去除異常值
#(data_train['DAYS_BIRTH'] / -365).describe() #正常
#data_train['DAYS_EMPLOYED'].describe() #異常

# Create an anomalous flag column
#data_train['DAYS_EMPLOYED_ANOM'] = data_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
data_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

#data_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
#plt.xlabel('Days Employment');

# for test data
#data_test['DAYS_EMPLOYED_ANOM'] = data_test["DAYS_EMPLOYED"] == 365243
data_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (data_test["DAYS_EMPLOYED_ANOM"].sum(), len(data_test)))

"""
#分析
data_train_1 = data_train.loc[data_train['TARGET'] == 1] #24825
data_train_0 = data_train.loc[data_train['TARGET'] == 0] #282686


#合約檢測
NAME = 'NAME_CONTRACT_TYPE'
sum(data_train_0[NAME]=='Revolving loans')
sum(data_train_0[NAME]=='Cash loans')
sum(data_train_1[NAME]=='Revolving loans')
sum(data_train_1[NAME]=='Cash loans')
sum(data_test[NAME]=='Revolving loans')
sum(data_test[NAME]=='Cash loans')

#性別檢測
NAME = 'CODE_GENDER'
sum(data_train_0[NAME]=='F')
sum(data_train_0[NAME]=='M')
sum(data_train_1[NAME]=='F')
sum(data_train_1[NAME]=='M')
sum(data_test[NAME]=='F')
sum(data_test[NAME]=='M')

#有無車檢測
NAME = 'FLAG_OWN_CAR'
sum(data_train_0[NAME]=='Y')
sum(data_train_0[NAME]=='N')
sum(data_train_1[NAME]=='Y')
sum(data_train_1[NAME]=='N')
sum(data_test[NAME]=='Y')
sum(data_test[NAME]=='N')

#有無房檢測
NAME = 'FLAG_OWN_REALTY'
sum(data_train_0[NAME]=='Y')
sum(data_train_0[NAME]=='N')
sum(data_train_1[NAME]=='Y')
sum(data_train_1[NAME]=='N')
sum(data_test[NAME]=='Y')
sum(data_test[NAME]=='N')

#有無子女檢測
NAME = 'CNT_CHILDREN'
data_train_0_sample = data_train_0.sample(25000)
sum(data_train_0[NAME]>0)/len(data_train_0)
sum(data_train_1[NAME]>0)/len(data_train_1)

# 年收入分析
#只分析年收入ＸＸＸ 以下者
NAME = 'AMT_INCOME_TOTAL'
INCOME_THRESHOLD = 2.5e+05
data_train_1 = data_train_1.loc[data_train_1[NAME] < INCOME_THRESHOLD]
data_train_0 = data_train_0.loc[data_train_0[NAME] < INCOME_THRESHOLD]
data_test_sample = data_test.loc[data_test[NAME] < INCOME_THRESHOLD]

data_train_0_sample = data_train_0.sample(25000)
data_test_sample = data_test_sample.sample(25000)

kde_target(NAME, data_train)

#信用點數
kde_target('AMT_CREDIT', data_train)

#貸款年金
kde_target('AMT_ANNUITY', data_train)

#貸款價錢
kde_target('AMT_GOODS_PRICE', data_train)


#誰陪伴辦理
NAME = 'NAME_TYPE_SUITE'
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_0[NAME]).plot.bar()
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_1[NAME]).plot.bar()

sum(data_train_0[NAME]=='Unaccompanied')/len(data_train_0)
sum(data_train_1[NAME]=='Unaccompanied')/len(data_train_1)

sum(data_train_0[NAME]=='Family')/len(data_train_0)
sum(data_train_1[NAME]=='Family')/len(data_train_1)

sum(data_train_0[NAME]=='Spouse, partner')/len(data_train_0)
sum(data_train_1[NAME]=='Spouse, partner')/len(data_train_1)

#收入種類
NAME = 'NAME_INCOME_TYPE'
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_0[NAME]).plot.bar()
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_1[NAME]).plot.bar()

sum(data_train_0[NAME]=='Working')/len(data_train_0)
sum(data_train_1[NAME]=='Working')/len(data_train_1)

sum(data_train_0[NAME]=='Commercial associate')/len(data_train_0)
sum(data_train_1[NAME]=='Commercial associate')/len(data_train_1)

sum(data_train_0[NAME]=='Pensioner')/len(data_train_0)
sum(data_train_1[NAME]=='Pensioner')/len(data_train_1)

sum(data_train_0[NAME]=='State servant')/len(data_train_0)
sum(data_train_1[NAME]=='State servant')/len(data_train_1)

#教育程度
NAME = 'NAME_EDUCATION_TYPE'
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_0[NAME]).plot.bar()
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_1[NAME]).plot.bar()

sum(data_train_0[NAME]=='Academic degree')/len(data_train_0)
sum(data_train_1[NAME]=='Academic degree')/len(data_train_1)


#婚姻狀況
NAME = 'NAME_FAMILY_STATUS'
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_0[NAME]).plot.bar()
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_1[NAME]).plot.bar()

#住處狀況
NAME = 'NAME_HOUSING_TYPE'
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_0[NAME]).plot.bar()
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_1[NAME]).plot.bar()


#居住城市人口 相關性不高
NAME = 'REGION_POPULATION_RELATIVE'
data_train_0[NAME].describe()
data_train_1[NAME].describe()


#天數
kde_target('DAYS_BIRTH', data_train)
kde_target('DAYS_EMPLOYED', data_train)
kde_target('DAYS_REGISTRATION', data_train)
kde_target('DAYS_ID_PUBLISH', data_train)


#OWN_CAR_AGE
kde_target('OWN_CAR_AGE', data_train)

#FLAG_MOBIL
#FLAG_EMP_PHONE
#FLAG_WORK_PHONE
#FLAG_CONT_MOBILE
#FLAG_PHONE
#FLAG_EMAIL

#OCCUPATION_TYPE
NAME = 'OCCUPATION_TYPE'
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_0[NAME]).plot.bar()
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_1[NAME]).plot.bar()

#REGION_RATING_CLIENT
NAME = 'REGION_RATING_CLIENT'
NAME = 'REGION_RATING_CLIENT_W_CITY'
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_0[NAME]).plot.bar()
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_1[NAME]).plot.bar()

#ORGANIZATION_TYPE
NAME = 'ORGANIZATION_TYPE'
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_0[NAME]).plot.bar()
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_1[NAME]).plot.bar()

#EXT_SOURCE
kde_target('EXT_SOURCE_1', data_train)
kde_target('EXT_SOURCE_2', data_train)
kde_target('EXT_SOURCE_3', data_train)

#不動產分析
kde_target('YEARS_BUILD_AVG', data_train)

#Doc use 3, 6
NAME = 'FLAG_DOCUMENT_21'
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_0[NAME]).plot.bar()
plt.figure(figsize = (8, 12))
pd.value_counts(data_train_1[NAME]).plot.bar()

#AMT_REQ_CREDIT_BUREAU
kde_target('AMT_REQ_CREDIT_BUREAU_QRT', data_train)

"""

#filter



#孩童個數
NAME = 'CNT_CHILDREN'
data_train.loc[data_train[NAME] > 3, NAME] = 'over_three'
data_train.replace({NAME:{0: 'zero', 1: 'one', 2: 'two', 3: 'three'}}, inplace = True)

#收入種類
NAME = 'NAME_INCOME_TYPE'
mask = (data_train[NAME] != 'Working') & \
        (data_train[NAME] != 'Commercial associate') & \
        (data_train[NAME] != 'Pensioner') & \
        (data_train[NAME] != 'State servant')
data_train.loc[mask, NAME] = 'others'

#時間相關
data_train['DAYS_BIRTH'] = data_train['DAYS_BIRTH'] / -365
data_train['DAYS_EMPLOYED'] = data_train['DAYS_EMPLOYED'] / -365
data_train['DAYS_REGISTRATION'] = data_train['DAYS_REGISTRATION'] / -365
data_train['DAYS_ID_PUBLISH'] = data_train['DAYS_ID_PUBLISH'] / -365

#AMT_REQ_CREDIT_BUREAU
req_count = np.zeros(data_train.shape[0])
req_count += data_train['AMT_REQ_CREDIT_BUREAU_HOUR']
req_count += data_train['AMT_REQ_CREDIT_BUREAU_DAY']
req_count += data_train['AMT_REQ_CREDIT_BUREAU_WEEK']
req_count += data_train['AMT_REQ_CREDIT_BUREAU_MON']
data_train['REQ_CNT'] = req_count

req_count = np.zeros(data_test.shape[0])
req_count += data_test['AMT_REQ_CREDIT_BUREAU_HOUR']
req_count += data_test['AMT_REQ_CREDIT_BUREAU_DAY']
req_count += data_test['AMT_REQ_CREDIT_BUREAU_WEEK']
req_count += data_test['AMT_REQ_CREDIT_BUREAU_MON']
data_test['REQ_CNT'] = req_count


data_train.drop(columns = useless_columns(), inplace=True)
data_test.drop(columns = useless_columns(), inplace=True)



#######################


bureau = pd.read_csv('bureau.csv') # (1716428, 17)


#https://www.kaggle.com/shanth84/home-credit-bureau-data-feature-engineering
B = bureau

#FEATURE 1 
grp = B[['SK_ID_CURR', 'DAYS_CREDIT']] \
  .groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT'] \
  .count() \
  .reset_index() \
  .rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')

#FEATURE 2
grp = B[['SK_ID_CURR', 'CREDIT_TYPE']] \
  .groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'] \
  .nunique() \
  .reset_index() \
  .rename(index=str, columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')

#FEATURE 3
B['AVERAGE_LOAN_TYPE'] = B['BUREAU_LOAN_COUNT']/B['BUREAU_LOAN_TYPES']

#FEATURE 4
B['CREDIT_ACTIVE_BINARY'] = B['CREDIT_ACTIVE']

def close_check(x):
    if x == 'Closed':
        y = 0
    else:
        y = 1    
    return y

B['CREDIT_ACTIVE_BINARY'] = B.apply(lambda x: close_check(x.CREDIT_ACTIVE), axis = 1)

grp = B.groupby(by = ['SK_ID_CURR'])['CREDIT_ACTIVE_BINARY'] \
  .mean() \
  .reset_index() \
  .rename(index=str, columns={'CREDIT_ACTIVE_BINARY': 'ACTIVE_LOANS_PERCENTAGE'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')
del B['CREDIT_ACTIVE_BINARY']
gc.collect()

#FEATURE 5
grp = B[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])
grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending = False)).reset_index(drop = True)#rename(index = str, columns = {'DAYS_CREDIT': 'DAYS_CREDIT_DIFF'})

grp1['DAYS_CREDIT1'] = grp1['DAYS_CREDIT']*-1
grp1['DAYS_DIFF'] = grp1.groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT1'].diff()
grp1['DAYS_DIFF'] = grp1['DAYS_DIFF'].fillna(0).astype('uint32')
del grp1['DAYS_CREDIT1'], grp1['DAYS_CREDIT'], grp1['SK_ID_CURR']
gc.collect()

B = B.merge(grp1, on = ['SK_ID_BUREAU'], how = 'left')

#FEATURE 6
B['CREDIT_ENDDATE_BINARY'] = B['DAYS_CREDIT_ENDDATE']

def ne_check(x):
    if x<0:
        y = 0
    else:
        y = 1   
    return y

B['CREDIT_ENDDATE_BINARY'] = B.apply(lambda x: ne_check(x.DAYS_CREDIT_ENDDATE), axis = 1)

grp = B.groupby(by = ['SK_ID_CURR'])['CREDIT_ENDDATE_BINARY'] \
  .mean() \
  .reset_index() \
  .rename(index=str, columns={'CREDIT_ENDDATE_BINARY': 'CREDIT_ENDDATE_PERCENTAGE'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')

#FEATURE 7
# We take only positive values of  ENDDATE since we are looking at Bureau Credit VALID IN FUTURE 
# as of the date of the customer's loan application with Home Credit 
B1 = B[B['CREDIT_ENDDATE_BINARY'] == 1]

B1['DAYS_CREDIT_ENDDATE1'] = B1['DAYS_CREDIT_ENDDATE']
# Groupby Each Customer ID 
grp = B1[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE1']].groupby(by = ['SK_ID_CURR'])
# Sort the values of CREDIT_ENDDATE for each customer ID 
grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT_ENDDATE1'], ascending = True)).reset_index(drop = True)
del grp
gc.collect()

# Calculate the Difference in ENDDATES and fill missing values with zero 
grp1['DAYS_ENDDATE_DIFF'] = grp1.groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE1'].diff()
grp1['DAYS_ENDDATE_DIFF'] = grp1['DAYS_ENDDATE_DIFF'].fillna(0).astype('uint32')
del grp1['DAYS_CREDIT_ENDDATE1'], grp1['SK_ID_CURR']
gc.collect()
print("Difference days calculated")

# Merge new feature 'DAYS_ENDDATE_DIFF' with original Data frame for BUREAU DATA
B = B.merge(grp1, on = ['SK_ID_BUREAU'], how = 'left')
del grp1
gc.collect()

# Calculate Average of DAYS_ENDDATE_DIFF

grp = B[['SK_ID_CURR', 'DAYS_ENDDATE_DIFF']].groupby(by = ['SK_ID_CURR'])['DAYS_ENDDATE_DIFF'].mean().reset_index().rename( index = str, columns = {'DAYS_ENDDATE_DIFF': 'AVG_ENDDATE_FUTURE'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')
del grp 
#del B['DAYS_ENDDATE_DIFF']
del B['CREDIT_ENDDATE_BINARY'], B['DAYS_CREDIT_ENDDATE']
gc.collect()

#FEATURE 8
B['AMT_CREDIT_SUM_DEBT'] = B['AMT_CREDIT_SUM_DEBT'].fillna(0)
B['AMT_CREDIT_SUM'] = B['AMT_CREDIT_SUM'].fillna(0)

grp1 = B[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
grp2 = B[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM': 'TOTAL_CUSTOMER_CREDIT'})

B = B.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
B = B.merge(grp2, on = ['SK_ID_CURR'], how = 'left')
del grp1, grp2
gc.collect()

B['DEBT_CREDIT_RATIO'] = B['TOTAL_CUSTOMER_DEBT']/B['TOTAL_CUSTOMER_CREDIT']

del B['TOTAL_CUSTOMER_DEBT'], B['TOTAL_CUSTOMER_CREDIT']
gc.collect()

#FEATURE 9
B['AMT_CREDIT_SUM_DEBT'] = B['AMT_CREDIT_SUM_DEBT'].fillna(0)
B['AMT_CREDIT_SUM_OVERDUE'] = B['AMT_CREDIT_SUM_OVERDUE'].fillna(0)

grp1 = B[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
grp2 = B[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})

B = B.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
B = B.merge(grp2, on = ['SK_ID_CURR'], how = 'left')
del grp1, grp2
gc.collect()

B['OVERDUE_DEBT_RATIO'] = B['TOTAL_CUSTOMER_OVERDUE']/(B['TOTAL_CUSTOMER_DEBT'] + B['TOTAL_CUSTOMER_OVERDUE'])

del B['TOTAL_CUSTOMER_OVERDUE'], B['TOTAL_CUSTOMER_DEBT']
gc.collect()

#FEATURE 10
B['CNT_CREDIT_PROLONG'] = B['CNT_CREDIT_PROLONG'].fillna(0)
grp = B[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']] \
  .groupby(by = ['SK_ID_CURR'])['CNT_CREDIT_PROLONG'] \
  .mean() \
  .reset_index() \
  .rename( index = str, columns = { 'CNT_CREDIT_PROLONG': 'AVG_CREDITDAYS_PROLONGED'})
B = B.merge(grp, on = ['SK_ID_CURR'], how = 'left')


new_columns = ['SK_ID_CURR',
               'BUREAU_LOAN_COUNT',
               'BUREAU_LOAN_TYPES',
               'AVERAGE_LOAN_TYPE',
               'ACTIVE_LOANS_PERCENTAGE',
               'CREDIT_ENDDATE_PERCENTAGE',
               'AVG_ENDDATE_FUTURE',
               'DEBT_CREDIT_RATIO',
               'OVERDUE_DEBT_RATIO',
               'AVG_CREDITDAYS_PROLONGED']

new_b = B[new_columns] \
  .groupby(by = ['SK_ID_CURR']) \
  .median() \
  .reset_index()


data_train = data_train.merge(new_b, on = ['SK_ID_CURR'], how = 'left')
data_test = data_test.merge(new_b, on = ['SK_ID_CURR'], how = 'left')


new_corrs = []

# Iterate through the columns 
for col in columns:
    # Calculate correlation with the target
    corr = data_train['TARGET'].corr(data_train[col])
    
    # Append the list as a tuple

    new_corrs.append((col, corr))
new_corrs = sorted(new_corrs, key = lambda x: abs(x[1]), reverse = True)
new_corrs[:15]



bureau_balance = pd.read_csv('bureau_balance.csv')
NAME = 'STATUS'
mask = (bureau_balance[NAME] != 'X') & \
        (bureau_balance[NAME] != 'C') & \
        (bureau_balance[NAME] != '0')
missed_payment = bureau_balance.loc[mask,'SK_ID_BUREAU']
missed_payment = pd.DataFrame({"SK_ID_BUREAU": missed_payment.unique(),
              "FLAG_MISSED":1})
bureau_missed_payment = bureau.merge(missed_payment, on = 'SK_ID_BUREAU', how = 'left')
bureau_missed_payment['FLAG_MISSED'] = bureau_missed_payment['FLAG_MISSED'].fillna(0)


bureau_missed_payment = bureau_missed_payment.groupby('SK_ID_CURR', as_index=False)['FLAG_MISSED'] \
  .max()


data_train = data_train.merge(bureau_missed_payment, on = 'SK_ID_CURR', how = 'left')
data_test = data_test.merge(bureau_missed_payment, on = 'SK_ID_CURR', how = 'left')




####################

prev = pd.read_csv('previous_application.csv')
prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
# Days 365.243 values -> nan
prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
# Add feature: value ask / value received percentage
prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
# Previous applications numeric features
num_aggregations = {
    'AMT_ANNUITY': ['min', 'max', 'mean'],
    'AMT_APPLICATION': ['min', 'max', 'mean'],
    'AMT_CREDIT': ['min', 'max', 'mean'],
    'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
    'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
    'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
    'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
    'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'CNT_PAYMENT': ['mean', 'sum'],
}
# Previous applications categorical features
cat_aggregations = {}
for cat in cat_cols:
    cat_aggregations[cat] = ['mean']

prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
# Previous Applications: Approved Applications - only numerical features
approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
cols = approved_agg.columns.tolist()
approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
# Previous Applications: Refused Applications - only numerical features
refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
del refused, refused_agg, approved, approved_agg, prev

for e in cols:
    prev_agg['NEW_RATIO_PREV_' + e[0] + "_" + e[1].upper()] = prev_agg['APPROVED_' + e[0] + "_" + e[1].upper()] / prev_agg['REFUSED_' + e[0] + "_" + e[1].upper()]

gc.collect()


data_train = data_train.merge(prev_agg, on = 'SK_ID_CURR', how = 'left')
data_test = data_test.merge(prev_agg, on = 'SK_ID_CURR', how = 'left')

####################

# Create a label encoder object, Ex:(Y,N) => (1,0)
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in data_train:
  if data_train[col].dtype == 'object':
    # If 2 or fewer unique categories
    if len(list(data_train[col].unique())) <= 2:
      print('Binary col name: ' + col)
      # Train on the training data
      le.fit(data_train[col])
      # Transform both training and testing data
      data_train[col] = le.transform(data_train[col])
      data_test[col] = le.transform(data_test[col])
      
      # Keep track of how many columns were label encoded
      le_count += 1
            
print('%d columns were label encoded.' % le_count)


# one-hot encoding of categorical variables
data_train, _ = one_hot_encoder(data_train)
data_test, _ = one_hot_encoder(data_test)


#移除test沒有 但是train卻有的冗余column
train_labels = data_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
data_train, data_test = data_train.align(data_test, join = 'inner', axis = 1)

# Add the target back in
data_train['TARGET'] = train_labels

print('Training Features shape: ', data_train.shape)
print('Testing Features shape: ', data_test.shape)




#####################




# Pearson correlation coefficient 
# Find correlations with the target and sort
correlations = data_train.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))



#####################

N_FOLDS = 5
k_fold = KFold(n_splits = N_FOLDS, shuffle = True, random_state = 50)

labels = data_train['TARGET']
features = data_train.drop(columns = ['SK_ID_CURR', 'TARGET'])
test_ids = data_test['SK_ID_CURR']
test_features = data_test.drop(columns = ['SK_ID_CURR'])

features = np.array(features)
test_features = np.array(test_features)

test_predictions = np.zeros(test_features.shape[0])
out_of_fold = np.zeros(features.shape[0])

valid_scores = []
train_scores = []

for train_indices, valid_indices in k_fold.split(features):
  
  # Training data for the fold
  train_features, train_labels = features[train_indices], labels[train_indices]
  # Validation data for the fold
  valid_features, valid_labels = features[valid_indices], labels[valid_indices]

  model = XGBClassifier()  
  model.fit(train_features, train_labels, eval_metric='auc',
            eval_set = [(train_features, train_labels), (valid_features, valid_labels)],
            early_stopping_rounds=300)
    
  test_predictions += model.predict_proba(test_features)[:, 1] / k_fold.n_splits
  out_of_fold[valid_indices] = model.predict_proba(valid_features)[:, 1]
  
  valid_score = max(model.evals_result()['validation_1']['auc'])
  train_score = max(model.evals_result()['validation_0']['auc'])
  
  valid_scores.append(valid_score)
  train_scores.append(train_score)
  
  # Clean up memory
  gc.enable()
  del model, train_features, valid_features
  gc.collect()
  
submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
submission.to_csv('baseline_xgb.csv', index = False)