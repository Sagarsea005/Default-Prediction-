#!/usr/bin/env python
# coding: utf-8

# # Project 27
# ####   The main aim of these project is to predict whether the customer will fall under default or not  

#Importing Libraries
import pandas as pd
import numpy as np
import pickle


#Importing Dataset
Bank=pd.read_csv('C:/Projects Data/bank_final.csv')


# ## Data Cleaning 

# Removing '$' and commas from records in columns with dollar values that should be floats
Bank[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']] = Bank[['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']].applymap(lambda x: x.strip().replace('$', '').replace(',', '')).astype(float)


# ## Feature Engineering 

Bank.loc[(Bank['FranchiseCode'] >= 1), 'FranchiseStatus'] = 1
Bank.loc[(Bank['FranchiseCode'] <1), 'FranchiseStatus'] = 0


# Remove records where RevLineCr != 'Y' or 'N' and LowDoc != 'Y' or 'N'
Bank = Bank[(Bank['RevLineCr'] == 'Y') | (Bank['RevLineCr'] == 'N')]
Bank = Bank[(Bank['LowDoc'] == 'Y') | (Bank['LowDoc'] == 'N')]

# RevLineCr and LowDoc: 0 = No, 1 = Yes
Bank['RevLineCr'] = np.where(Bank['RevLineCr'] == 'N', 0, 1)
Bank['LowDoc'] = np.where(Bank['LowDoc'] == 'N', 0, 1)


#  Converting MIS_Status to numeric format
Bank['MIS_Status'] = np.where(Bank['MIS_Status'] == 'P I F', 1, 0)
Bank['MIS_Status'].value_counts()


df=Bank.drop(['Name','City','State','Bank','BankState','FranchiseCode','Zip','CCSC','ApprovalDate','ApprovalFY','DisbursementDate','BalanceGross','ChgOffDate','ChgOffPrinGr','GrAppv','SBA_Appv'],axis=1)


df['FranchiseStatus']=df['FranchiseStatus'].astype('int64')


df = df[['Term','NoEmp','CreateJob','RetainedJob','UrbanRural','FranchiseStatus','NewExist','RevLineCr','LowDoc','DisbursementGross','MIS_Status']]


# ## Train & Test Split
XFinal_Features=df[['Term','NoEmp','CreateJob','RetainedJob','UrbanRural','FranchiseStatus','NewExist','RevLineCr','LowDoc','DisbursementGross']]
YFinal_Features=df['MIS_Status']


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(XFinal_Features,YFinal_Features,test_size=0.20)


# ### Model Building 

from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(bootstrap=True,criterion='entropy',max_depth= 10, max_features= 8, min_samples_leaf=1, n_estimators=10,random_state = 25)


rfc.fit(X_train, y_train)


# Saving model to disk
pickle.dump(rfc, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

