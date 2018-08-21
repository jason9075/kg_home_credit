#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:13:19 2018

@author: jason9075
"""
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_corr(df):
  new_corrs = []

  # Iterate through the columns 
  for col in df.columns:
      # Calculate correlation with the target
      corr = df['TARGET'].corr(df[col])
      
      # Append the list as a tuple
  
      new_corrs.append((col, corr))
  new_corrs = sorted(new_corrs, key = lambda x: abs(x[1]), reverse = True)
  return new_corrs


def missing_values_table(df):
  # Total missing values
  mis_val = df.isnull().sum()
  
  # Percentage of missing values
  mis_val_percent = 100 * df.isnull().sum() / len(df)
  
  # Make a table with the results
  mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
  
  # Rename the columns
  mis_val_table_ren_columns = mis_val_table.rename(
  columns = {0 : 'Missing Values', 1 : '% of Total Values'})
  
  # Sort the table by percentage of missing descending
  mis_val_table_ren_columns = mis_val_table_ren_columns[
      mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
  '% of Total Values', ascending=False).round(1)
  
  # Print some summary information
  print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
      "There are " + str(mis_val_table_ren_columns.shape[0]) +
        " columns that have missing values.")
  
  # Return the dataframe with missing information
  return mis_val_table_ren_columns

def add_score(df, column, low_bound=0, low_bound_weight=0, high_bound=1, high_bound_weight=0, other=0):   
  value = df[column]
  value = value/(value.max() - value.min())
  value = value.apply(lambda x: low_bound_weight*x if x < low_bound else high_bound_weight*x if high_bound < x else other if (np.isnan(x)==False) else 0) 
  return value

def kde_target(var_name, df):
    
    # Calculate the correlation coefficient between the new variable and the target
    corr = df['TARGET'].corr(df[var_name])
    
    # Calculate medians for repaid vs not repaid
    avg_repaid = df.ix[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.ix[df['TARGET'] == 1, var_name].median()
    
    plt.figure(figsize = (12, 6))
    
    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.ix[df['TARGET'] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.ix[df['TARGET'] == 1, var_name], label = 'TARGET == 1')
    
    # label the plot
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend();
    
    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)
    
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
  
  
def prev_extract(prev_df, prev_type, is_drop_date=False):
  # Cash loans 
  prev_cash_loans = prev_df[prev_df['NAME_CONTRACT_TYPE']=='Cash loans']
  prev_cash_loans.drop(columns=['NAME_CONTRACT_TYPE',
                                'NAME_GOODS_CATEGORY',
                                'NAME_CONTRACT_STATUS'], axis=1, inplace= True)
  # remove 4 cash value
  prev_cash_loans = prev_cash_loans[prev_cash_loans['PRODUCT_COMBINATION']!='Cash']
  prev_cash_loans, cat_cols = one_hot_encoder(prev_cash_loans, nan_as_category= False)
  
  if is_drop_date:
    num_aggregations = {
      'AMT_ANNUITY': ['mean','sum'],#-0.0386, -0.0342
      'AMT_APPLICATION': ['mean','sum'], #-0.0313, -0.0269
      'AMT_CREDIT': ['mean','sum'], #-0.0261, -0.0230
      'CNT_PAYMENT': ['mean', 'max'], #0.0428, 0.0366
      'RATIO_ANN_TO_APP':['mean'] #-0.0046
    }  
  else:
    num_aggregations = {
      'AMT_ANNUITY': ['mean','sum'],#-0.0386, -0.0342
      'AMT_APPLICATION': ['mean','sum'], #-0.0313, -0.0269
      'AMT_CREDIT': ['mean','sum'], #-0.0261, -0.0230
      'DAYS_LAST_DUE_DIFF':['mean'],
      'DAYS_LAST_DUE_1ST_VERSION_DIFF':['mean'],
      'DAYS_TERMINATION_DIFF':['mean'],
      'CNT_PAYMENT': ['mean', 'max'], #0.0428, 0.0366
      'RATIO_ANN_TO_APP':['mean'] #-0.0046
    }
  
  cat_aggregations = {}
  for cat in cat_cols:
    cat_aggregations[cat] = ['mean', 'sum']
    
  prev_agg = prev_cash_loans.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
  prev_agg.columns = pd.Index(['PREV_'+prev_type+'_CASH_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
  prev_agg['PREV_'+prev_type+'_CASH_COUNT'] = prev_cash_loans.groupby('SK_ID_CURR').size()

  del prev_cash_loans
  
  # Consumer loans
  prev_consumer_loans = prev_df[prev_df['NAME_CONTRACT_TYPE']=='Consumer loans']
  
  prev_consumer_loans.drop(columns=['NAME_CONTRACT_TYPE',
                                    'NAME_CONTRACT_STATUS'], axis=1, inplace= True)
  prev_consumer_loans, cat_cols = one_hot_encoder(prev_consumer_loans, nan_as_category= False)
  
  if is_drop_date:
    num_aggregations = {
      'AMT_ANNUITY': ['mean','sum'],#-0.0470, -0.0567
      'AMT_APPLICATION': ['mean','sum'], #-0.0376, -0.0507
      'AMT_CREDIT': ['mean','sum'],#-0.0328, -0.0471
      'RATE_DOWN_PAYMENT': ['mean','max'], #-0.0323, -0.0422
      'CNT_PAYMENT': ['mean', 'max'], #0.0135,0.0002
      'RATIO_ANN_TO_APP':['mean'] #0.0116
    }
  else:
    num_aggregations = {
      'AMT_ANNUITY': ['mean','sum'],#-0.0470, -0.0567
      'AMT_APPLICATION': ['mean','sum'], #-0.0376, -0.0507
      'AMT_CREDIT': ['mean','sum'],#-0.0328, -0.0471
      'RATE_DOWN_PAYMENT': ['mean','max'], #-0.0323, -0.0422
      'DAYS_LAST_DUE_DIFF':['mean'],
      'DAYS_LAST_DUE_1ST_VERSION_DIFF':['mean'],
      'DAYS_TERMINATION_DIFF':['mean'],
      'CNT_PAYMENT': ['mean', 'max'], #0.0135,0.0002
      'RATIO_ANN_TO_APP':['mean'] #0.0116
    }
  
  cat_aggregations = {}
  for cat in cat_cols:
    cat_aggregations[cat] = ['mean']
    
  consume_agg = prev_consumer_loans.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
  consume_agg.columns = pd.Index(['PREV_'+prev_type+'_CONSUME_' + e[0] + "_" + e[1].upper() for e in consume_agg.columns.tolist()])
  prev_agg = prev_agg.join(consume_agg, how='outer', on='SK_ID_CURR')
  prev_agg['PREV_'+prev_type+'_CONSUME_COUNT'] = prev_consumer_loans.groupby('SK_ID_CURR').size()

  del prev_consumer_loans, consume_agg
  
  
  # Revolving loans
  prev_revolving_loans = prev_df[prev_df['NAME_CONTRACT_TYPE']=='Revolving loans']
  prev_revolving_loans = prev_revolving_loans[prev_revolving_loans['AMT_CREDIT']!=0]

  #revolving_loans 有些AMT_APPLICATION是零 改用AMT_CREDIT
  prev_revolving_loans['RATIO_ANN_TO_APP'] = prev_revolving_loans['AMT_ANNUITY'] / prev_revolving_loans['AMT_CREDIT']

  prev_revolving_loans.drop(columns=['NAME_CONTRACT_TYPE',
                                     'NAME_GOODS_CATEGORY',
                                     'NAME_CONTRACT_STATUS'], axis=1, inplace= True)
  prev_revolving_loans, cat_cols = one_hot_encoder(prev_revolving_loans, nan_as_category= False)
  
  if is_drop_date:
    num_aggregations = {
      'AMT_ANNUITY': ['mean','sum'], #-0.0246, -0.0228
      'AMT_CREDIT': ['mean','sum'],#-0.0242, -0.0225
      'RATIO_ANN_TO_APP':['mean'] #-0.0082
    }
  else:
    num_aggregations = {
      'AMT_ANNUITY': ['mean','sum'], #-0.0246, -0.0228
      'AMT_CREDIT': ['mean','sum'],#-0.0242, -0.0225
      'DAYS_LAST_DUE_DIFF':['mean'],
      'DAYS_TERMINATION_DIFF':['mean'],
      'RATIO_ANN_TO_APP':['mean'] #-0.0082
    }
    
  cat_aggregations = {}
  for cat in cat_cols:
    cat_aggregations[cat] = ['mean']
    
  revolving_agg = prev_revolving_loans.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
  revolving_agg.columns = pd.Index(['PREV_'+prev_type+'_REVOLVING_' + e[0] + "_" + e[1].upper() for e in revolving_agg.columns.tolist()])
  prev_agg = prev_agg.join(revolving_agg, how='outer', on='SK_ID_CURR')
  prev_agg['PREV_'+prev_type+'_REVOLVING_COUNT'] = prev_revolving_loans.groupby('SK_ID_CURR').size()

  del prev_revolving_loans, revolving_agg, prev_df
  
  return prev_agg
  
def app_useless_columns():
    return ['NAME_TYPE_SUITE', 
                'NAME_FAMILY_STATUS',
                'NAME_HOUSING_TYPE',
                'REGION_POPULATION_RELATIVE',
                'FLAG_MOBIL',
                'FLAG_EMP_PHONE',
                'FLAG_WORK_PHONE',
                'FLAG_CONT_MOBILE',
                'FLAG_PHONE',
                'FLAG_OWN_CAR',
                'FLAG_EMAIL',
                'CNT_FAM_MEMBERS',
                'REGION_RATING_CLIENT',
                'REGION_RATING_CLIENT_W_CITY',
                'WEEKDAY_APPR_PROCESS_START',
                'HOUR_APPR_PROCESS_START',
                'REG_REGION_NOT_LIVE_REGION',
                'REG_REGION_NOT_WORK_REGION',
                'LIVE_REGION_NOT_WORK_REGION',
                'REG_CITY_NOT_LIVE_CITY',
                'REG_CITY_NOT_WORK_CITY',
                'LIVE_CITY_NOT_WORK_CITY',
                'BASEMENTAREA_AVG',
                'YEARS_BEGINEXPLUATATION_AVG',
                'YEARS_BUILD_AVG',
                'COMMONAREA_AVG',
                'ELEVATORS_AVG',
                'ENTRANCES_AVG',
                'FLOORSMAX_AVG',
                'FLOORSMIN_AVG',
                'LANDAREA_AVG',
                'LIVINGAPARTMENTS_AVG',
                'LIVINGAREA_AVG',
                'NONLIVINGAPARTMENTS_AVG',
                'NONLIVINGAREA_AVG',
                'APARTMENTS_MODE',
                'APARTMENTS_AVG',
                'BASEMENTAREA_MODE',
                'YEARS_BEGINEXPLUATATION_MODE',
                'YEARS_BUILD_MODE',
                'COMMONAREA_MODE',
                'ELEVATORS_MODE',
                'ENTRANCES_MODE',
                'FLOORSMAX_MODE',
                'FLOORSMIN_MODE',
                'LANDAREA_MODE',
                'LIVINGAPARTMENTS_MODE',
                'LIVINGAREA_MODE',
                'NONLIVINGAPARTMENTS_MODE',
                'NONLIVINGAREA_MODE',
                'APARTMENTS_MEDI',
                'BASEMENTAREA_MEDI',
                'YEARS_BEGINEXPLUATATION_MEDI',
                'YEARS_BUILD_MEDI',
                'COMMONAREA_MEDI',
                'ELEVATORS_MEDI',
                'ENTRANCES_MEDI',
                'FLOORSMAX_MEDI',
                'FLOORSMIN_MEDI',
                'LANDAREA_MEDI',
                'LIVINGAPARTMENTS_MEDI',
                'LIVINGAREA_MEDI',
                'NONLIVINGAPARTMENTS_MEDI',
                'NONLIVINGAREA_MEDI',
                'FONDKAPREMONT_MODE',
                'HOUSETYPE_MODE',
                'TOTALAREA_MODE',
                'WALLSMATERIAL_MODE',
                'EMERGENCYSTATE_MODE',
                'DAYS_LAST_PHONE_CHANGE',
                'FLAG_DOCUMENT_2',
                'FLAG_DOCUMENT_4',
                'FLAG_DOCUMENT_5',
                'FLAG_DOCUMENT_7',
                'FLAG_DOCUMENT_8',
                'FLAG_DOCUMENT_9',
                'FLAG_DOCUMENT_10',
                'FLAG_DOCUMENT_11',
                'FLAG_DOCUMENT_12',
                'FLAG_DOCUMENT_13',
                'FLAG_DOCUMENT_14',
                'FLAG_DOCUMENT_15',
                'FLAG_DOCUMENT_16',
                'FLAG_DOCUMENT_17',
                'FLAG_DOCUMENT_18',
                'FLAG_DOCUMENT_19',
                'FLAG_DOCUMENT_20',
                'FLAG_DOCUMENT_21',
                'AMT_REQ_CREDIT_BUREAU_HOUR',
                'AMT_REQ_CREDIT_BUREAU_DAY',
                'AMT_REQ_CREDIT_BUREAU_WEEK',
                'AMT_REQ_CREDIT_BUREAU_MON',
                'AMT_REQ_CREDIT_BUREAU_QRT',
                'AMT_REQ_CREDIT_BUREAU_YEAR']
    
def app_option_columns():
    return ['CNT_CHILDREN',
                'ORGANIZATION_TYPE',
                'OCCUPATION_TYPE',
                'FLAG_DOCUMENT_3',
                'FLAG_DOCUMENT_6']
    
  
    
def bureau_useless_columns():
    return ['BURO_CREDIT_TYPE_nan_MEAN',
                'BURO_CREDIT_ACTIVE_nan_MEAN']

    
def bureau_optional_columns():
    return ['BURO_AMT_CREDIT_MAX_OVERDUE_MEAN',
                'BURO_AMT_ANNUITY_MAX', 
                'BURO_AMT_ANNUITY_MEAN',
                'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN',
                'BURO_AMT_CREDIT_SUM_MAX',
                'BURO_AMT_CREDIT_SUM_MEAN',
                'BURO_AMT_CREDIT_SUM_SUM',
                'BURO_AMT_CREDIT_SUM_LIMIT_MEAN',
                'BURO_CREDIT_ACTIVE_Bad debt_MEAN',
                'BURO_AMT_CREDIT_SUM_LIMIT_SUM',
                'BURO_CREDIT_ACTIVE_Bad debt_MEAN',
                'BURO_AMT_CREDIT_SUM_OVERDUE_MEAN']

def prev_useless_columns():
  return ['PREV_NAME_CONTRACT_TYPE_nan_MEAN',
          'PREV_NAME_CONTRACT_STATUS_nan_MEAN',
          'PREV_NAME_CLIENT_TYPE_nan_MEAN',
          'PREV_NAME_GOODS_CATEGORY_nan_MEAN',
          'PREV_PRODUCT_COMBINATION_nan_MEAN']
  