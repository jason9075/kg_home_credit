#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 16:15:07 2018

@author: jason9075
"""

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import util
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False, epison=1):
  df = pd.read_pickle('application_train_test_cached')
  return df
  # Read data and merge
  df = pd.read_csv('application_train.csv', nrows= num_rows)  
  test_df = pd.read_csv('application_test.csv', nrows= num_rows)
  print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
  df = df.append(test_df).reset_index()
  # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
  df = df[df['CODE_GENDER'] != 'XNA']
      
  # NaN values for DAYS_EMPLOYED: 365.243 -> nan
  df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
  
  #孩童個數
# =============================================================================
#   NAME = 'CNT_CHILDREN'
#   df.loc[df[NAME] > 3, NAME] = 'over_three'
#   df.replace({NAME:{0: 'zero', 1: 'one', 2: 'two', 3: 'three'}}, inplace = True)
# 
# =============================================================================
  #收入種類
  NAME = 'NAME_INCOME_TYPE'
  mask = (df[NAME] != 'Working') & \
          (df[NAME] != 'Commercial associate') & \
          (df[NAME] != 'Pensioner') & \
          (df[NAME] != 'State servant')
  df.loc[mask, NAME] = 'others'
  
  #AMT_REQ_CREDIT_BUREAU
  req_count = np.zeros(df.shape[0])
  req_count += df['AMT_REQ_CREDIT_BUREAU_HOUR']
  req_count += df['AMT_REQ_CREDIT_BUREAU_DAY']
  req_count += df['AMT_REQ_CREDIT_BUREAU_WEEK']
  req_count += df['AMT_REQ_CREDIT_BUREAU_MON']
  df['REQ_CNT'] = req_count

  df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'])
  df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE']+ epison)
  df['NEW_CREDIT_TO_INCOME'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL']+ epison)
  df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] - epison)
  df['NEW_SOURCES_PROD_12'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
  df['NEW_SOURCES_PROD_23'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
  df['NEW_SOURCES_PROD_13'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
  df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
  df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
  df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
  df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (df['DAYS_BIRTH'] - epison)
  df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_BIRTH'] - epison)
  
  df['DEF_TO_OBS_30_RATIO'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE']
  df['DEF_TO_OBS_60_RATIO'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['OBS_60_CNT_SOCIAL_CIRCLE']
  
  # Categorical features with Binary encode (0 or 1; two categories)
  for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
      df[bin_feature], uniques = pd.factorize(df[bin_feature])
      
  #移除不明顯columns
  df.drop(columns = util.app_useless_columns(), inplace=True)
  df.drop(columns = util.app_option_columns(), inplace=True)

  # Categorical features with One-Hot encode
  df, cat_cols = util.one_hot_encoder(df, nan_as_category)

  del test_df, req_count
  gc.collect()
  
  df.drop(columns=['NAME_EDUCATION_TYPE_Academic degree',
                     'NAME_INCOME_TYPE_Pensioner',
                     'NAME_INCOME_TYPE_others',
                     'NAME_INCOME_TYPE_nan',
                     'NAME_EDUCATION_TYPE_nan',
                     'NAME_INCOME_TYPE_nan',
                     'NAME_CONTRACT_TYPE_nan'], axis=1, inplace= True)

  
# =============================================================================
#   #cached
#   df.to_pickle('application_train_test_cached')
# 
# =============================================================================
  
  return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True, epison=1):
  bb_agg = pd.read_pickle('bureau_cached')
  return bb_agg
  bureau = pd.read_csv('bureau.csv')
  bb = pd.read_csv('bureau_balance.csv')
  
  #TYPE
  NAME = 'CREDIT_TYPE'
  mask = (bureau[NAME] == 'Consumer credit') | (bureau[NAME] == 'Credit card')
  bureau = bureau.loc[mask]
  
  #DAYS_CREDIT_ENDDATE
  NAME = 'DAYS_CREDIT_ENDDATE'
  mask = bureau[NAME] < -200000
  bureau.loc[mask,NAME] = np.nan
  
  #DAYS_ENDDATE_FACT
  NAME = 'DAYS_ENDDATE_FACT'
  mask = bureau[NAME] < -4000
  bureau.loc[mask,NAME] = np.nan
  
  #AMT_CREDIT_MAX_OVERDUE
  NAME = 'AMT_CREDIT_MAX_OVERDUE'
  mask = bureau[NAME] > 10000000
  bureau.loc[mask,NAME] = np.nan
  
  #AMT_CREDIT_SUM
  NAME = 'AMT_CREDIT_SUM'
  mask = bureau[NAME] > 1000
  bureau = bureau.loc[mask]
  mask = bureau[NAME] > 50000000
  bureau.loc[mask,NAME] = np.nan
  
  #AMT_CREDIT_SUM_DEBT
  NAME = 'AMT_CREDIT_SUM_DEBT'
  mask = bureau[NAME] < 0
  bureau.loc[mask,NAME] = 0
  
  #AMT_CREDIT_SUM_LIMIT
  NAME = 'AMT_CREDIT_SUM_LIMIT'
  mask = bureau[NAME] < 0
  bureau.loc[mask,NAME] = 0
  
  #AMT_ANNUITY
  NAME = 'AMT_ANNUITY'
  mask = (bureau[NAME] > 10000000) | (bureau[NAME] < 100)
  bureau.loc[mask,NAME] = np.nan

  
  #收入種類
  NAME = 'CREDIT_TYPE'
  mask = (bureau[NAME] != 'Consumer credit') & \
          (bureau[NAME] != 'Credit card') & \
          (bureau[NAME] != 'Mortgage') & \
          (bureau[NAME] != 'Car loan') & \
          (bureau[NAME] != 'Microloan')
  bureau.loc[mask, NAME] = 'Others'
  
  bureau['ENDDATE_DIFF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT']
  bureau['ENDDATE_FACT_DIFF'] = bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT']

  bureau.drop(columns=['CREDIT_CURRENCY'], axis=1, inplace= True)

  # Bureau balance: Perform aggregations and merge with bureau.csv
  NAME = 'STATUS'
  mask = (bb[NAME] != 'C') & (bb[NAME] != 'X')
  bb = bb[mask]
  
  bb['DPD_SCORE'] = (1.2)**bb['MONTHS_BALANCE']
  bb['DPD_SCORE'] = (bb['STATUS']!='0') * bb['DPD_SCORE']
  bb_dpd = bb.groupby(by = ['SK_ID_BUREAU'])['DPD_SCORE'] \
              .sum()

  bureau = bureau.join(bb_dpd, how='left', on='SK_ID_BUREAU')

  bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
  del bb, bb_dpd
  gc.collect()
   
  bureau, bureau_cat = util.one_hot_encoder(bureau, nan_as_category)

  # Bureau and bureau_balance numeric features
  num_aggregations = {
    'ENDDATE_DIFF': ['mean'],
    'ENDDATE_FACT_DIFF': ['mean'],
    'AMT_CREDIT_MAX_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'sum'],
    'AMT_ANNUITY': ['max', 'mean'],
    'DPD_SCORE': ['mean']
  }
  # Bureau and bureau_balance categorical features
  cat_aggregations = {}
  for cat in bureau_cat: cat_aggregations[cat] = ['mean']
  
  bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
  bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
  
  
  
  # Bureau: Active credits - using only numerical aggregations
  active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
  
  num_aggregations = {
    'ENDDATE_DIFF': ['mean'],
    'AMT_CREDIT_MAX_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'sum']
  }
  
  active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
  cols = active_agg.columns.tolist()
  active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
  bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
  del active, active_agg
  gc.collect()
  
  
  # Bureau: Closed credits - using only numerical aggregations
  closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
  
  num_aggregations = {
    'ENDDATE_DIFF': ['mean'],
    'ENDDATE_FACT_DIFF': ['mean'],
    'AMT_CREDIT_MAX_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM': ['max', 'mean','sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM_LIMIT': ['max', 'mean', 'sum'],
    'AMT_ANNUITY': ['max', 'mean'],
    'AMT_CREDIT_SUM_DEBT': ['mean', 'sum']
  }
  
  closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
  closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
  bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
  
  for e in cols:
    bureau_agg['NEW_RATIO_BURO_' + e[0] + "_" + e[1].upper()] = bureau_agg['ACTIVE_' + e[0] + "_" + e[1].upper()] / (bureau_agg['CLOSED_' + e[0] + "_" + e[1].upper()]+epison)

  #移除不明顯columns
  bureau_agg.drop(columns = util.bureau_useless_columns(), inplace=True)
  bureau_agg.drop(columns = util.bureau_optional_columns(), inplace=True)

  bureau_agg['BUREAU_SCORE'] = 0
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'ACTIVE_AMT_CREDIT_SUM_MEAN',
        0.4, 0.4, 0.7, -0.2)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'BURO_CREDIT_ACTIVE_Closed_MEAN',
        0.3, 0.1, 0.8, -0.1)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'ACTIVE_AMT_CREDIT_SUM_SUM',
        high_bound=0.7, high_bound_weight=0.4)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'BURO_AMT_CREDIT_SUM_DEBT_SUM',
        high_bound=0.7, high_bound_weight=0.3)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'ACTIVE_ENDDATE_DIFF_MEAN',
        low_bound=0.15, low_bound_weight=-0.1)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN',
        high_bound=0.7, high_bound_weight=0.5)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'BURO_AMT_CREDIT_SUM_DEBT_MAX',
        0.3, 0, 0.7, -0.1, 0.2)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'ACTIVE_AMT_CREDIT_SUM_MAX',
        0.3, 0, 0.7, -0.2, 0.1)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'BURO_ENDDATE_FACT_DIFF_MEAN',
        high_bound=0.7, high_bound_weight=0.25)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'BURO_CREDIT_ACTIVE_Sold_MEAN',
        high_bound=0.7, high_bound_weight=0.15)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN',
        0.1, -0.2, 0.7, 0.3)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN',
        0.2, -0.1, 0.7, 0.1)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN',
        high_bound=0.8, high_bound_weight=-0.2)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'CLOSED_AMT_CREDIT_SUM_MEAN',
        low_bound=0.3, low_bound_weight=0.21)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'CLOSED_AMT_CREDIT_SUM_MAX',
        high_bound=0.8, high_bound_weight=-0.1)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'NEW_RATIO_BURO_ENDDATE_DIFF_MEAN',
        high_bound=0.8, high_bound_weight=-0.2)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'ACTIVE_AMT_CREDIT_SUM_DEBT_SUM',
        0.2, -0.2, 0.7, 0.3)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'CLOSED_AMT_ANNUITY_MAX',
        0.3, -0.1, 0.8, 0.1)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'NEW_RATIO_BURO_AMT_CREDIT_MAX_OVERDUE_MEAN',
        high_bound=0.85, high_bound_weight=0.6)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'NEW_RATIO_BURO_AMT_CREDIT_SUM_SUM',
        low_bound=0.2, low_bound_weight=-0.15)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_MEAN',
        high_bound=0.85, high_bound_weight=-0.2)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM',
        high_bound=0.8, high_bound_weight=0.35)
  bureau_agg['BUREAU_SCORE'] += util.add_score(bureau_agg, 
        'BURO_DPD_SCORE_MEAN',
        high_bound=0.7, high_bound_weight=0.5)
  
  bureau_agg.drop(columns=['CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN',
                     'CLOSED_AMT_CREDIT_SUM_LIMIT_SUM',
                     'NEW_RATIO_BURO_AMT_CREDIT_SUM_OVERDUE_MEAN'], axis=1, inplace= True)

  del closed, closed_agg, bureau
  gc.collect()
  
# =============================================================================
#   #cached
#   bureau_agg.to_pickle('bureau_cached')
# =============================================================================
  
  return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
  prev_agg = pd.read_pickle('prev_cached')
  return prev_agg
  prev = pd.read_csv('previous_application.csv', nrows = num_rows)
  
  # Optional: Remove 346 XNA NAME_CONTRACT_TYPE 
  prev = prev[prev['NAME_CONTRACT_TYPE'] != 'XNA']
  
  # Optional: Remove useless columns 
  prev.drop(columns=['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT', 
                     'NFLAG_LAST_APPL_IN_DAY','NAME_CASH_LOAN_PURPOSE','DAYS_DECISION',
                     'NAME_PAYMENT_TYPE','NAME_SELLER_INDUSTRY','NAME_TYPE_SUITE',
                     'NAME_PORTFOLIO','NAME_PRODUCT_TYPE','CHANNEL_TYPE',
                     'NAME_YIELD_GROUP','SELLERPLACE_AREA','RATE_INTEREST_PRIMARY',
                     'RATE_INTEREST_PRIVILEGED','CODE_REJECT_REASON',
                     'NAME_CLIENT_TYPE'], axis=1, inplace= True)

  #AMT_DOWN_PAYMENT
  NAME = 'AMT_DOWN_PAYMENT'
  mask = prev[NAME] < 0
  prev.loc[mask, NAME] = 0
  
  #NAME_GOODS_CATEGORY
  NAME = 'NAME_GOODS_CATEGORY'
  mask = (prev[NAME] != 'XNA') & \
          (prev[NAME] != 'Mobile') & \
          (prev[NAME] != 'Consumer Electronics') & \
          (prev[NAME] != 'Computers') & \
          (prev[NAME] != 'Audio/Video') & \
          (prev[NAME] != 'Furniture') & \
          (prev[NAME] != 'Photo / Cinema Equipment') & \
          (prev[NAME] != 'Construction Materials') & \
          (prev[NAME] != 'Clothing and Accessories') & \
          (prev[NAME] != 'Auto Accessories') & \
          (prev[NAME] != 'Jewelry') & \
          (prev[NAME] != 'Homewares') & \
          (prev[NAME] != 'Medical Supplies') & \
          (prev[NAME] != 'Vehicles')
  prev.loc[mask, NAME] = 'Others'
  
  # Days 365.243 values -> nan
  prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
  prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
  prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
  prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
  prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
  
  # Add feature: value ask / value received percentage
  prev['DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE'] - prev['DAYS_FIRST_DUE']
  prev['DAYS_LAST_DUE_1ST_VERSION_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_FIRST_DUE']
  prev['DAYS_TERMINATION_DIFF'] = prev['DAYS_TERMINATION'] - prev['DAYS_FIRST_DUE']
  prev['RATIO_ANN_TO_APP'] = prev['AMT_ANNUITY'] / prev['AMT_APPLICATION']

  # Previous Applications:
  prev_approved = prev[prev['NAME_CONTRACT_STATUS']=='Approved']
  prev_refused = prev[prev['NAME_CONTRACT_STATUS']=='Refused']
  
  prev_agg = util.prev_extract(prev_approved,'APPR')
  prev_refused_agg = util.prev_extract(prev_refused,'REFU',is_drop_date=True)
  prev_agg = prev_agg.join(prev_refused_agg, how='outer', on='SK_ID_CURR')

  # cash
  prev_agg['PREV_CASH_SCORE'] = 0
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CASH_DAYS_LAST_DUE_1ST_VERSION_DIFF_MEAN',
          0.3, -0.25, 0.6, 0.5)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CASH_PRODUCT_COMBINATION_Cash X-Sell: low_MEAN',
          high_bound=0.8, high_bound_weight=-0.2)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CASH_CNT_PAYMENT_MAX',
          high_bound=0.8, high_bound_weight=0.2)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CASH_CNT_PAYMENT_MEAN',
          high_bound=0.9, high_bound_weight=0.15)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CASH_PRODUCT_COMBINATION_Cash Street: high_MEAN',
          high_bound=0.7, high_bound_weight=0.2)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CASH_DAYS_LAST_DUE_DIFF_MEAN',
          high_bound=0.7, high_bound_weight=-0.15)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CASH_AMT_CREDIT_SUM',
          high_bound=0.7, high_bound_weight=0.3)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CASH_PRODUCT_COMBINATION_Cash X-Sell: high_MEAN',
          high_bound=0.7, high_bound_weight=0.1)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CASH_PRODUCT_COMBINATION_Cash Street: low_MEAN',
          high_bound=0.9, high_bound_weight=-0.22)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CASH_AMT_ANNUITY_MEAN',
          0.2, 0.1, 0.7, -0.1)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CASH_AMT_ANNUITY_SUM',
          0.2, 0.1, 0.7, -0.1)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CASH_AMT_APPLICATION_SUM',
          high_bound=0.8, high_bound_weight=0.15)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CASH_PRODUCT_COMBINATION_Cash X-Sell: low_MEAN',
          low_bound=0.2, high_bound_weight=0.7)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CASH_PRODUCT_COMBINATION_Cash Street: low_MEAN',
          0.2, 0.15, 0.8, -0.15)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CASH_CNT_PAYMENT_MEAN',
          high_bound=0.6, high_bound_weight=0.16)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CONSUME_AMT_ANNUITY_MEAN',
          low_bound=0.2, low_bound_weight=0.1)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CASH_PRODUCT_COMBINATION_Cash Street: high_MEAN',
          high_bound=0.7, high_bound_weight=0.1)
  prev_agg['PREV_CASH_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CONSUME_AMT_CREDIT_SUM',
          low_bound=0.2, low_bound_weight=-0.1)

          
  #consume 
  prev_agg['PREV_CONSUME_SCORE'] = 0
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CONSUME_CNT_PAYMENT_MAX',
          high_bound=0.7, high_bound_weight=0.2)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CONSUME_AMT_APPLICATION_MEAN',
          low_bound=0.2, low_bound_weight=0.15)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CONSUME_AMT_APPLICATION_SUM',
          0.2, 0.1, 0.8, -0.07)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CASH_CNT_PAYMENT_MEAN',
          high_bound=0.6, high_bound_weight=0.13)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CASH_AMT_ANNUITY_MEAN',
          0.2, 0.1, 0.8, -0.1)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CONSUME_NAME_GOODS_CATEGORY_Furniture_MEAN',
          high_bound=0.8, high_bound_weight=-0.1)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CONSUME_NAME_GOODS_CATEGORY_Auto Accessories_MEAN',
          low_bound=0.2, low_bound_weight=-0.05)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CONSUME_RATE_DOWN_PAYMENT_MAX',
          high_bound=0.6, high_bound_weight=-0.1)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CONSUME_RATE_DOWN_PAYMENT_MEAN',
          high_bound=0.7, high_bound_weight=-0.05)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
        'PREV_APPR_CONSUME_PRODUCT_COMBINATION_POS household without interest_MEAN',
        high_bound=0.7, high_bound_weight=-0.07)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CONSUME_PRODUCT_COMBINATION_POS mobile with interest_MEAN',
          high_bound=0.7, high_bound_weight=0.05)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CONSUME_DAYS_LAST_DUE_1ST_VERSION_DIFF_MEAN',
          high_bound=0.8, high_bound_weight=0.15)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CONSUME_NAME_GOODS_CATEGORY_Photo / Cinema Equipment_MEAN',
          high_bound=0.8, high_bound_weight=-0.1)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CONSUME_NAME_GOODS_CATEGORY_Audio/Video_MEAN',
          high_bound=0.8, high_bound_weight=0.1)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_REVOLVING_AMT_ANNUITY_SUM',
          high_bound=0.7, high_bound_weight=-0.2)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_REVOLVING_AMT_ANNUITY_MEAN',
          low_bound=0.2, low_bound_weight=0.1)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_APPR_CONSUME_COUNT',
          high_bound=0.8, high_bound_weight=0.03)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CONSUME_RATE_DOWN_PAYMENT_MAX',
          low_bound=0.3, low_bound_weight=0.08)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CONSUME_RATE_DOWN_PAYMENT_MEAN',
          0.3, 0.08, 0.8, -0.06)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CONSUME_AMT_APPLICATION_MEAN',
          low_bound=0.2, low_bound_weight=0.07)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CONSUME_CNT_PAYMENT_MEAN',
          low_bound=0.2, high_bound_weight=-0.05)
  prev_agg['PREV_CONSUME_SCORE'] += util.add_score(prev_agg, 
          'PREV_REFU_CONSUME_CNT_PAYMENT_MAX',
          0.2, -0.04, 0.8, -0.02)
  
  prev_agg.drop(columns=['PREV_APPR_CONSUME_NAME_GOODS_CATEGORY_Homewares_MEAN',
                     'PREV_APPR_CONSUME_NAME_GOODS_CATEGORY_Medical Supplies_MEAN',
                     'PREV_APPR_CONSUME_NAME_GOODS_CATEGORY_XNA_MEAN',
                     'PREV_APPR_REVOLVING_COUNT',
                     'PREV_REFU_CONSUME_NAME_GOODS_CATEGORY_Furniture_MEAN',
                     'PREV_REFU_CONSUME_NAME_GOODS_CATEGORY_Others_MEAN',
                     'PREV_REFU_CONSUME_NAME_GOODS_CATEGORY_Vehicles_MEAN',
                     'PREV_REFU_CONSUME_NAME_GOODS_CATEGORY_Jewelry_MEAN',
                     'PREV_REFU_CONSUME_NAME_GOODS_CATEGORY_Homewares_MEAN',
                     'PREV_REFU_CONSUME_NAME_GOODS_CATEGORY_Clothing and Accessories_MEAN',
                     'PREV_REFU_CONSUME_PRODUCT_COMBINATION_POS other with interest_MEAN',
                     'PREV_REFU_CONSUME_NAME_GOODS_CATEGORY_Auto Accessories_MEAN',
                     'PREV_REFU_CONSUME_PRODUCT_COMBINATION_POS industry without interest_MEAN',
                     'PREV_REFU_CONSUME_NAME_GOODS_CATEGORY_Medical Supplies_MEAN',
                     'PREV_REFU_CONSUME_PRODUCT_COMBINATION_POS others without interest_MEAN',
                     'PREV_REFU_CONSUME_NAME_GOODS_CATEGORY_Construction Materials_MEAN',
                     'PREV_APPR_CONSUME_PRODUCT_COMBINATION_POS others without interest_MEAN',
                     'PREV_REFU_CONSUME_PRODUCT_COMBINATION_POS household without interest_MEAN',
                     'PREV_APPR_CASH_PRODUCT_COMBINATION_Cash Street: middle_SUM',
                     'PREV_APPR_REVOLVING_RATIO_ANN_TO_APP_MEAN',
                     'PREV_REFU_CASH_PRODUCT_COMBINATION_Cash Street: low_SUM',
                     'PREV_REFU_CASH_PRODUCT_COMBINATION_Cash X-Sell: low_SUM',
                     'PREV_REFU_CASH_PRODUCT_COMBINATION_Cash X-Sell: middle_SUM',
                     'PREV_APPR_REVOLVING_DAYS_TERMINATION_DIFF_MEAN'], axis=1, inplace= True)


        
  del prev, prev_approved, prev_refused
# =============================================================================
#   plt.figure(figsize = (8, 12))
#   pd.value_counts(df['NAME_CONTRACT_TYPE']).plot.bar()  
#   pd.value_counts(prev_revolving_loans['NAME_GOODS_CATEGORY']).plot.bar()
#   prev_revolving_loans = prev_revolving_loans.merge(df[['TARGET','SK_ID_CURR']], how='left', on='SK_ID_CURR')
#   prev, cat_cols = util.one_hot_encoder(prev, nan_as_category= True)
# =============================================================================
  gc.collect()
  
# =============================================================================
#   #cached
#   prev_agg.to_pickle('prev_cached')
# =============================================================================
  
  return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
  pos_agg = pd.read_pickle('pos_cached')
  return pos_agg
  
  pos = pd.read_csv('POS_CASH_balance.csv', nrows = num_rows)
      
  pos_act_status = pos.groupby('SK_ID_CURR')['SK_ID_PREV'] \
                    .nunique() \
                    .to_frame() \
                    .rename(columns={'SK_ID_PREV': 'POS_CNT_PREV_ID'})
    
  pos_latest_status = pos.sort_values('MONTHS_BALANCE', ascending=False).groupby(['SK_ID_CURR','SK_ID_PREV']) \
                        .first()
  pos_latest_status.index = pos_latest_status.index.get_level_values('SK_ID_CURR')
                        
  pos_complete_count = pos_latest_status \
                        .groupby('SK_ID_CURR') \
                        .apply(lambda x: (x['NAME_CONTRACT_STATUS']=='Completed').sum())\
                        .rename('POS_CNT_COMPLETED')
  
  pos_active_count = pos_latest_status\
                        .groupby('SK_ID_CURR') \
                        .apply(lambda x: ((x['NAME_CONTRACT_STATUS']=='Active') & (0 < x['CNT_INSTALMENT_FUTURE'])).sum()) \
                        .rename('POS_CNT_ACTIVED') 
  
  pos_act = pos[pos['NAME_CONTRACT_STATUS']=='Active']
  
  aggregations = {
      'SK_ID_CURR':['mean'],
      'SK_DPD': ['max'],
      'SK_DPD_DEF': ['max']
  }
  pos_act = pos_act.groupby('SK_ID_PREV') \
                        .agg(aggregations) \
                        .reset_index(drop=True)
  pos_act.columns = ['SK_ID_CURR', 'SK_DPD_MAX', 'SK_DPD_DEF_MAX']
  pos_act['DPD_RATIO'] = pos_act['SK_DPD_MAX'] / pos_act['SK_DPD_DEF_MAX']
  
  aggregations = {
      'SK_DPD_MAX': ['max', 'mean'],
      'SK_DPD_DEF_MAX': ['max', 'mean'],
      'DPD_RATIO':['max', 'mean']
  }
  pos_agg = pos_act.groupby('SK_ID_CURR') \
            .agg(aggregations)
  
  pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
                      
  pos_act_status = pos_act_status.join(pos_agg, how='left', on='SK_ID_CURR')
  pos_act_status = pos_act_status.join(pos_complete_count, how='left', on='SK_ID_CURR')
  pos_act_status = pos_act_status.join(pos_active_count, how='left', on='SK_ID_CURR')
  pos_act_status['POS_FLAG_UNCOMPLETED'] = (pos_act_status['POS_CNT_COMPLETED'] < pos_act_status['POS_CNT_PREV_ID']).astype(int)
  pos_act_status['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
  
  pos_act_status['POS_SCORE'] = 0
  pos_act_status['POS_SCORE'] += util.add_score(pos_act_status, 
                      'POS_SK_DPD_DEF_MAX_MEAN',
                      high_bound=0.8, high_bound_weight=0.6)
  pos_act_status['POS_SCORE'] += util.add_score(pos_act_status, 
                      'POS_COUNT',
                      0.3, 0.2, 0.6, -0.1)
  pos_act_status['POS_SCORE'] += util.add_score(pos_act_status, 
                      'POS_SK_DPD_MAX_MEAN',
                      high_bound=0.8, high_bound_weight=0.2)
  
  del pos, pos_complete_count, pos_active_count, pos_agg, pos_act, pos_latest_status
  gc.collect()
# =============================================================================
#   #cached
#   pos_act_status.to_pickle('pos_cached')
# =============================================================================
  
  return pos_act_status
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True, epison = 1e-03):
  ins_agg = pd.read_pickle('ins_cached')
  return ins_agg
  
  ins = pd.read_csv('installments_payments.csv', nrows = num_rows)
  
  #mix same period AMT_PAYMENT ex:100013
  num_aggregations = {
      'AMT_PAYMENT': ['sum'],
      'DAYS_ENTRY_PAYMENT': ['min']
  }
  ins = ins.groupby(['SK_ID_CURR','NUM_INSTALMENT_VERSION','NUM_INSTALMENT_NUMBER','DAYS_INSTALMENT','AMT_INSTALMENT']) \
            .agg({**num_aggregations}).reset_index()
            
  ins.columns = pd.Index([e[0] for e in ins.columns.tolist()])
  
  #去除特殊情況者
  mask = (ins['AMT_INSTALMENT']==67.5) & (ins['DAYS_ENTRY_PAYMENT'].isnull())
  ins = ins.loc[~mask]

  #去除最近要繳錢者與費用過低者
  ins = ins[ins['DAYS_INSTALMENT']<-180]
  ins = ins[ins['AMT_INSTALMENT']>1000]
              
  
  # Percentage and difference paid in each installment (amount paid and installment value)
  ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / (ins['AMT_INSTALMENT'] + epison)
  ins['PAYMENT_PERC'] = ins['PAYMENT_PERC'].apply(lambda x: x if x < 1 else 1)
  # Days past due and days before due (no negative values)
  ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
  ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
  ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
  ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)    
  
  # Features: Perform aggregations
  aggregations = {
    'NUM_INSTALMENT_VERSION': ['nunique'],
    'DPD': ['max', 'mean', 'sum'],
    'DBD': ['max', 'mean', 'sum'],
    'PAYMENT_PERC': ['mean'],
    'AMT_INSTALMENT': ['max', 'mean', 'sum'],
    'AMT_PAYMENT': ['min', 'max', 'mean', 'sum']
  }

  ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
  ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
  # Count installments accounts
  ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
  ins_agg_dpd_count = ins.groupby('SK_ID_CURR') ['DPD']\
                                              .apply(lambda x: (0<x).sum())
                                              
  ins_agg_dbd_count = ins.groupby('SK_ID_CURR') ['DBD']\
                                              .apply(lambda x: (0<x).sum())
                                              
  ins_agg['INSTAL_DPD_DBD_RATIO'] = ins_agg_dpd_count / (ins_agg_dbd_count + epison)
  ins_agg['INSTAL_DPD_DBD_DIFF'] = ins_agg_dpd_count - ins_agg_dbd_count


  ins_agg['INSTAL_SCORE'] = 0
  ins_agg['INSTAL_SCORE'] += util.add_score(ins_agg, 
          'INSTAL_DBD_MAX',
          low_bound=0.2, low_bound_weight=-0.2)
  ins_agg['INSTAL_SCORE'] += util.add_score(ins_agg, 
          'INSTAL_AMT_PAYMENT_SUM',
          0.2, 0.2, 0.8, -0.2)
  ins_agg['INSTAL_SCORE'] += util.add_score(ins_agg, 
          'INSTAL_DPD_MEAN',
          high_bound=0.8, high_bound_weight=0.2)
  ins_agg['INSTAL_SCORE'] += util.add_score(ins_agg, 
          'INSTAL_DPD_DBD_DIFF',
          0.1, -0.1, 0.6, 0.15)
  ins_agg['INSTAL_SCORE'] += util.add_score(ins_agg, 
          'INSTAL_COUNT',
          high_bound=0.8, high_bound_weight=-0.3)
  ins_agg['INSTAL_SCORE'] += util.add_score(ins_agg, 
          'INSTAL_DBD_MEAN',
          high_bound=0.8, high_bound_weight=-0.15)
  ins_agg['INSTAL_SCORE'] += util.add_score(ins_agg, 
          'INSTAL_DBD_SUM',
          low_bound=0.2, low_bound_weight=0.25)
  ins_agg['INSTAL_SCORE'] += util.add_score(ins_agg, 
          'INSTAL_PAYMENT_PERC_MEAN',
          0.2, 0.18, 0.8, -0.1)
  ins_agg['INSTAL_SCORE'] += util.add_score(ins_agg, 
          'INSTAL_AMT_PAYMENT_MIN',
          high_bound=0.8, high_bound_weight=-0.15)
  ins_agg['INSTAL_SCORE'] += util.add_score(ins_agg, 
          'INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE',
          high_bound=0.7, high_bound_weight=-0.2)
  ins_agg['INSTAL_SCORE'] += util.add_score(ins_agg, 
          'INSTAL_AMT_INSTALMENT_MEAN',
          high_bound=0.8, high_bound_weight=0.15)
  ins_agg['INSTAL_SCORE'] += util.add_score(ins_agg, 
          'INSTAL_DPD_MAX',
          high_bound=0.8, high_bound_weight=0.15)


  del ins, ins_agg_dpd_count, ins_agg_dbd_count
  gc.collect()
  
# =============================================================================
#   #cached
#   ins_agg.to_pickle('ins_cached')
# =============================================================================
  
  return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
  cc_agg = pd.read_pickle('cc_cached')
  return cc_agg
  cc = pd.read_csv('credit_card_balance.csv', nrows = num_rows)
  cc.drop(['SK_ID_PREV','AMT_DRAWINGS_ATM_CURRENT',
            'AMT_DRAWINGS_OTHER_CURRENT','AMT_DRAWINGS_POS_CURRENT',
            'CNT_DRAWINGS_ATM_CURRENT','CNT_DRAWINGS_CURRENT','CNT_DRAWINGS_OTHER_CURRENT',
            'CNT_DRAWINGS_POS_CURRENT','CNT_INSTALMENT_MATURE_CUM',
            'NAME_CONTRACT_STATUS'], axis= 1, inplace = True)

  #add column
  cc['FLAG_EXCEED_LIMIT'] = cc['AMT_CREDIT_LIMIT_ACTUAL'] < cc['AMT_BALANCE']
  cc['AMT_PAYMENT_MAX'] = cc[['AMT_PAYMENT_CURRENT','AMT_PAYMENT_TOTAL_CURRENT']].apply(max, axis=1)

  #Raw
  aggregations = {
      'AMT_BALANCE': ['max', 'mean'],
      'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
      'AMT_INST_MIN_REGULARITY': ['max'],
      'AMT_PAYMENT_CURRENT': ['mean'],
      'AMT_PAYMENT_TOTAL_CURRENT': ['max', 'mean'],
      'AMT_PAYMENT_MAX': ['mean'],
      'AMT_TOTAL_RECEIVABLE': ['max', 'mean']
  }
  
  cc_agg = cc.groupby('SK_ID_CURR')\
            .agg(aggregations)
  cc_agg.columns = pd.Index(['CC_RAW_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

  #remove redundant data
  aggregations = {
    'AMT_BALANCE': ['mean'],
    'AMT_INST_MIN_REGULARITY': ['mean'],
    'AMT_PAYMENT_CURRENT': ['sum'],
    'AMT_TOTAL_RECEIVABLE': ['mean']
  }

  cc_remove_unuse_agg = cc.groupby('SK_ID_CURR') \
                      .filter(lambda x: x['AMT_BALANCE'].max() > 0.) \
                      .reset_index() \
                      .groupby('SK_ID_CURR') \
                      .agg(aggregations)

  cc_remove_unuse_agg.columns = pd.Index(['CC_RM_UNUSE_' + e[0] + "_" + e[1].upper() for e in cc_remove_unuse_agg.columns.tolist()])
    
  #exceed zero
  aggregations = {
    'AMT_BALANCE': ['mean'],
    'AMT_DRAWINGS_CURRENT': ['max'],
    'AMT_PAYMENT_CURRENT': ['sum'],
  }   
  
  cc_exceed_zero_agg = cc[cc['AMT_BALANCE']>0].groupby('SK_ID_CURR')\
                                          .agg(aggregations)
  cc_exceed_zero_agg.columns = pd.Index(['CC_EXCEED_' + e[0] + "_" + e[1].upper() for e in cc_exceed_zero_agg.columns.tolist()])

  # General aggregations
  cc_agg = cc_agg.join(cc_remove_unuse_agg, how='left', on='SK_ID_CURR')   
  cc_agg = cc_agg.join(cc_exceed_zero_agg, how='left', on='SK_ID_CURR')    

  # Count credit card lines
  cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
  
  cc_agg['CC_SCORE'] = 0
  cc_agg['CC_SCORE'] += util.add_score(cc_agg, 
          'CC_RAW_AMT_DRAWINGS_CURRENT_MAX',
          low_bound=0.2, low_bound_weight=-0.12)
  cc_agg['CC_SCORE'] += util.add_score(cc_agg, 
          'CC_RAW_AMT_DRAWINGS_CURRENT_MEAN',
          high_bound=0.6, high_bound_weight=0.3)
  cc_agg['CC_SCORE'] += util.add_score(cc_agg, 
          'CC_RAW_AMT_BALANCE_MEAN',
          high_bound=0.8, high_bound_weight=0.2)
  cc_agg['CC_SCORE'] += util.add_score(cc_agg, 
          'CC_RM_UNUSE_AMT_PAYMENT_CURRENT_SUM',
          low_bound=0.4, low_bound_weight=0.3)
  cc_agg['CC_SCORE'] += util.add_score(cc_agg, 
          'CC_EXCEED_AMT_PAYMENT_CURRENT_SUM',
          low_bound=0.2, low_bound_weight=0.15)
  cc_agg['CC_SCORE'] += util.add_score(cc_agg, 
          'CC_EXCEED_AMT_BALANCE_MEAN',
          high_bound=0.8, high_bound_weight=0.2)
  cc_agg['CC_SCORE'] += util.add_score(cc_agg, 
          'CC_RAW_AMT_PAYMENT_TOTAL_CURRENT_MEAN',
          low_bound=0.2, low_bound_weight=0.1)
  cc_agg['CC_SCORE'] += util.add_score(cc_agg, 
          'CC_RAW_AMT_PAYMENT_TOTAL_CURRENT_MAX',
          high_bound=0.75, high_bound_weight=0.2)
  cc_agg['CC_SCORE'] += util.add_score(cc_agg, 
          'CC_COUNT',
          high_bound=0.9, high_bound_weight=0.1)
  cc_agg['CC_SCORE'] += util.add_score(cc_agg, 
          'CC_RAW_AMT_BALANCE_MAX',
          high_bound=0.85, high_bound_weight=0.1)
  cc_agg['CC_SCORE'] += util.add_score(cc_agg, 
          'CC_RM_UNUSE_AMT_TOTAL_RECEIVABLE_MEAN',
          high_bound=0.85, high_bound_weight=0.1)
  cc_agg['CC_SCORE'] += util.add_score(cc_agg, 
          'CC_RAW_AMT_DRAWINGS_CURRENT_SUM',
          high_bound=0.85, high_bound_weight=0.1)
  cc_agg['CC_SCORE'] += util.add_score(cc_agg, 
          'CC_RAW_AMT_TOTAL_RECEIVABLE_MAX',
          high_bound=0.85, high_bound_weight=0.15)
  
  
  del cc, cc_remove_unuse_agg, cc_exceed_zero_agg
  gc.collect()
  
# =============================================================================
#   #cached
#   cc_agg.to_pickle('cc_cached')
# =============================================================================
  
  return cc_agg
  
def mix_columns(df):
  #NEW_EXT_SOURCES_MEAN = 0.22
  
  #PREV
  #df["MIX_PREV_CASH_ANNUITY_MEAN_TO_INCOME"] = df["PREV_APPR_CONSUME_AMT_APPLICATION_SUM"] / df["AMT_INCOME_TOTAL"] #0.0945

  
  #CC
  df["MIX_CC_BAL_MEAN_TO_INCOME"] = df["CC_RAW_AMT_BALANCE_MEAN"] / df["AMT_INCOME_TOTAL"] #0.0945
  df["MIX_CC_BAL_MAX_TO_INCOME"] = df["CC_RAW_AMT_BALANCE_MAX"] / df["AMT_INCOME_TOTAL"] #0.0759
  df["MIX_CC_TOTAL_RECEIVABLE_MEAN_TO_INCOME"] = df["CC_RAW_AMT_TOTAL_RECEIVABLE_MEAN"] / df["AMT_INCOME_TOTAL"] #0.0938

# =============================================================================
#   df["test"] = df["INSTAL_AMT_PAYMENT_MEAN"] / (df["AMT_INCOME_TOTAL"] + 1) #0.0938
#   
#   util.kde_target('MIX_CC_BAL_MEAN_TO_INCOME', df) 
# =============================================================================
  return df


# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lgb(df, num_folds, debug= False, file_name='default'):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LGB. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    valid_scores = []
    train_scores = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
      dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx], 
                             label=train_df['TARGET'].iloc[train_idx], 
                             free_raw_data=False, silent=True)
      dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx], 
                           label=train_df['TARGET'].iloc[valid_idx], 
                           free_raw_data=False, silent=True)
      
      # LightGBM parameters found by Bayesian optimization
      params = {
          'objective': 'binary',
          'boosting_type': 'gbdt',
          'nthread': 4,
          'learning_rate': 0.02,  # 02,
          'num_leaves': 20,
          'colsample_bytree': 0.9497036,
          'subsample': 0.8715623,
          'subsample_freq': 1,
          'max_depth': 8,
          'reg_alpha': 0.041545473,
          'reg_lambda': 0.0735294,
          'min_split_gain': 0.0222415,
          'min_child_weight': 60, # 39.3259775,
          'seed': 0,
          'verbose': -1,
          'metric': 'auc',
      }
      
      clf = lgb.train(
          params=params,
          train_set=dtrain,
          num_boost_round=10000,
          valid_sets=[dtrain, dvalid],
          early_stopping_rounds=200,
          verbose_eval=False
      )

      oof_preds[valid_idx] = clf.predict(dvalid.data)
      sub_preds += clf.predict(test_df[feats]) / folds.n_splits
      
      valid_score = clf.best_score['valid_1']['auc']
      train_score = clf.best_score['training']['auc']
      
      valid_scores.append(valid_score)
      train_scores.append(train_score)
      
      fold_importance_df = pd.DataFrame()
      fold_importance_df["feature"] = feats
      fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
      fold_importance_df["fold"] = n_fold + 1
      feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
      print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(dvalid.label, oof_preds[valid_idx])))
      del clf, dtrain, dvalid
      gc.collect()
      
      
    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
      sub_df = test_df[['SK_ID_CURR']].copy()
      sub_df['TARGET'] = sub_preds
      sub_df[['SK_ID_CURR', 'TARGET']].to_csv(file_name, index= False)
    feature_importance_df.to_csv('feature_importance_df.csv', index= False)
    display_importances(feature_importance_df)
    #return feature_importance_df
    return (valid_scores, train_scores)
    
def kfold_xgb(df, num_folds, debug= False, file_name='default'):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting XGB. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    valid_scores = []
    train_scores = []

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
      # Training data for the fold
      train_features, train_labels = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
      # Validation data for the fold
      valid_features, valid_labels = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

      model = XGBClassifier()  
      
      model.fit(train_features, train_labels, eval_metric='auc',
          eval_set = [(train_features, train_labels), (valid_features, valid_labels)],
          early_stopping_rounds=300)
      


      oof_preds[valid_idx] = model.predict_proba(valid_features)[:, 1]
      sub_preds += model.predict_proba(test_df[feats])[:, 1] / folds.n_splits

      valid_score = max(model.evals_result()['validation_1']['auc'])
      train_score = max(model.evals_result()['validation_0']['auc'])
      
      valid_scores.append(valid_score)
      train_scores.append(train_score)
      
      fold_importance_df = pd.DataFrame()
      fold_importance_df["feature"] = feats
      fold_importance_df["fold"] = n_fold + 1
      print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_labels, oof_preds[valid_idx])))
      del train_features, train_labels, valid_features, valid_labels
      gc.collect()
      
    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        sub_df = test_df[['SK_ID_CURR']].copy()
        sub_df['TARGET'] = sub_preds
        sub_df[['SK_ID_CURR', 'TARGET']].to_csv(file_name, index= False)
    return (valid_scores, train_scores)

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(12, 8))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('Features (avg over folds)')
    plt.tight_layout
    plt.savefig('importances01.png')
    best_features.to_csv('best_features.csv', index= False)



def main(debug = False, bureau_enable=True, prev_enable=True, 
         pos_enable=True,ins_enable=True, cc_enable=True, 
         file_name='default.csv'):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    if bureau_enable:
      with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    if prev_enable:
      with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    if pos_enable:
      with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    if ins_enable:
      with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    if cc_enable:
      with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("Mix columns"):
      df = mix_columns(df)
      print(df.shape)
      gc.collect()
    with timer("Run train with kfold"):
      print(df.shape)
      gc.collect()
      print(df.shape)
      score = kfold_lgb(df, num_folds= 5, debug= debug, file_name=file_name)
      print("val score: ", score[0], "\n train score: ", score[1])

if __name__ == "__main__":
    submission_file_name = "submission_with_lgb_cc.csv"
    with timer("Full model run"):
      main(debug = False, bureau_enable=True, prev_enable=True, 
           pos_enable=True, ins_enable=True, cc_enable=True, 
           file_name='submission_with_lgb_score.csv')
      
      main(debug = False, bureau_enable=False, prev_enable=False, 
           pos_enable=False, ins_enable=False, cc_enable=False, 
           file_name='submission_with_lgb_none.csv')
      main(debug = False, bureau_enable=True, prev_enable=False, 
             pos_enable=False, ins_enable=False, cc_enable=False, 
             file_name='submission_with_lgb_bureau.csv')
      main(debug = False, bureau_enable=False, prev_enable=True, 
             pos_enable=False, ins_enable=False, cc_enable=False, 
             file_name='submission_with_lgb_prev.csv')
      main(debug = False, bureau_enable=False, prev_enable=False, 
             pos_enable=True, ins_enable=False, cc_enable=False, 
             file_name='submission_with_lgb_pos.csv')
      main(debug = False, bureau_enable=False, prev_enable=False, 
             pos_enable=False, ins_enable=True, cc_enable=False, 
             file_name='submission_with_lgb_ins.csv')
      main(debug = False, bureau_enable=False, prev_enable=False, 
             pos_enable=False, ins_enable=False, cc_enable=True, 
             file_name='submission_with_lgb_cc.csv')
