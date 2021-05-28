# -*- coding: utf-8 -*-
"""
Created on Sun May 16 11:50:09 2021

@author: manoranjan.parida
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import re
from pyspark import SparkContext
import pyspark.sql 
from pyspark.sql import SQLContext
import sqlite3

os.getcwd()
os.chdir(r'C:\Users\Komali Srinivas\Day1\capstone')
data_cust= pd.read_excel('Customer_Data.xlsx')
data_FI = pd.read_csv('Final_invoice.csv')
data_JTD = pd.read_csv('JTD.csv')
data_Plant= pd.read_excel('Plant Master.xlsx')

#data_cust.head()
#data_cust.shape
#data_FI.head()
#data_FI.shape
#data_JTD.head()
#data_JTD.shape
#data_Plant.head()
#data_Plant.shape

# Renaming Column Names
data_cust.columns = data_cust.columns.str.replace(' ', '')
data_cust.columns = data_cust.columns.str.replace('.', '')
data_cust.columns
data_FI.columns = data_FI.columns.str.replace(' ', '')
data_FI.columns = data_FI.columns.str.replace('%', '')
data_FI.columns = data_FI.columns.str.replace('(', '_')
data_FI.columns = data_FI.columns.str.replace(')', '')
data_FI.columns = data_FI.columns.str.replace('/', '_')
data_FI.columns = data_FI.columns.str.replace('.', '_')
data_FI.columns = data_FI.columns.str.replace(':', '_')
data_FI.columns
data_JTD.columns = data_JTD.columns.str.replace(' ', '')
data_JTD.columns = data_JTD.columns.str.replace(':', '_')
data_JTD.columns
data_Plant.columns = data_Plant.columns.str.replace(' ', '')
data_Plant.columns = data_Plant.columns.str.replace('.-', '')
data_Plant.columns

# Creating Data Base

#1-->Customers total spend based on Model
#2-->Plant Revenue based on Geography location
#3-->Spare/Part Utilization Usage Pattern
#4-->Seasonal Surge



conn = sqlite3.connect('ThreePointers_Capstone.db')
#data_Plant.to_sql("PLANT_MASTER", conn, if_exists = "replace")
#data_cust.to_sql("Customer_Master", conn, if_exists = "replace")
#data_FI.to_sql("Final_Invoice", conn, if_exists = "replace")
#data_JTD.to_sql("JTD", conn, if_exists = "replace")

cur = conn.cursor()
conn = sqlite3.connect('ThreePointers_Capstone.db')
Cust_Model_Spend = pd.DataFrame(cur.execute("SELECT * FROM Cust_Model_Spend"))


data_JTD.columns

data_FI.columns

data_FI['ODN No.'].unique()

Numeric_data_FI=data_FI.select_dtypes(exclude='object')

data_FI.columns.type

for i in Numeric_data_FI:
    data_FI[i].plot(kind='scatter')
    plt.show()

data_FI['KMs Reading'].describe()
np.quantile(data_FI['KMs Reading'],[0.68,0.96,0.98])

data_FI['KMs Reading']
data_FI['District']

plt.figure(figsize = (10,20))
sns.lineplot(data=data_FI, x="District", y="Usage")
plt.xticks(rotation=90)
plt.show()

data_FI['Usage'] = 0

data_FI['Usage'][data_FI['KMs Reading']>=70000]= '1'
data_FI['Usage'][(data_FI['KMs Reading']>=40000) | (data_FI['KMs Reading']<70000)]= '2'
data_FI['Usage'][data_FI['KMs Reading']<40000]= '3'

data_FI['Usage'].value_counts()

data_FI['District'].unique()

#sns.barplot(x="District", y=data_FI['Usage'].count())


data_Plant[]

Plant_break = pd.read_csv('plantdet.csv')

Plant_break['TotalValue'].describe()

#for totalvalue vs state
plt.figure(figsize = (20,20))
sns.barplot(data=Plant_break, x="State", y="TotalValue")
plt.xticks(rotation=90)
plt.show()

#make wise total value
plt.figure(figsize = (20,20))
sns.barplot(data=Plant_break, x="Make", y="TotalValue")
plt.xticks(rotation=90)
plt.show()


Plant_break.info()

Numeric_data_pb=Plant_break.select_dtypes(exclude='object')
Cat_data_pb=Plant_break.select_dtypes(include='object')

Plant_break.columns

test = pd.merge(data_JTD, Plant_break,  how='left', left_on='DBM Order', right_on = 'JobCardNo')


data_JTD['DBM Order'] == Plant_break['JobCardNo']

fin_date = pd.read_csv('fin_data.csv')

fin_date.info()

model_revenue = fin_date.groupby(['Make'])['TotalValue'].sum().reset_index()

model_revenue.sort_values('TotalValue',ascending=False, inplace = True)

#make wise total value
plt.figure(figsize = (20,20))
sns.barplot(data=model_revenue, x="Make", y="TotalValue")
plt.xticks(rotation=90)
plt.show()


datasets=[fin_date]
columns_to_keep=[]
columns_to_drop=[]
fin_date.name = "Merged_Data_FI_JTD_Plant"

for j in datasets:
    temp_sep="--> Dataset Picked of DatFrame <--"+j.name
    columns_to_keep.append(temp_sep)
    columns_to_drop.append(temp_sep)
    for i in j.columns:
        per_missing_Data=(j.shape[0] - j[i].isnull().sum())/j.shape[0]>.35
        if per_missing_Data==True:
            columns_to_keep.append(i)
        else:
            columns_to_drop.append(i)

print(columns_to_keep)
print('*'*100)
print(columns_to_drop)

fin_date.info()

Numeric_data_pb=fin_date.select_dtypes(exclude='object')
Cat_data_pb=fin_date.select_dtypes(include='object')

Cat_data_pb.info()

Numeric_data_pb['ServiceAdvisorName'].value_counts()

Numeric_data_pb.drop(columns = ['index:1','Unnamed_0','CGST_14', 'CGST_2_5','CGST_6', 'CGST_9', 'IGST_12', 'IGST_18', 'IGST_28', 'IGST_5','SGST_UGST_14', 'SGST_UGST_2_5', 'SGST_UGST_6','SGST_UGST_9', 'ServiceAdvisorName', 'TDSamount', 'TotalAmtWtdTax_',
       'SGST_UGST_9','Unnamed_0:1', 'TotalCGST', 'TotalGST', 'TotalIGST', 'TotalSGST_UGST','index:2', 'Unnamed_0:1'],inplace = True)


Numeric_data_pb_drop = Numeric_data_pb[['index', 'index:1','Unnamed_0','CGST_14', 'CGST_2_5','CGST_6', 'CGST_9', 'IGST_12', 'IGST_18', 'IGST_28', 'IGST_5','SGST_UGST_14', 'SGST_UGST_2_5', 'SGST_UGST_6','SGST_UGST_9', 'ServiceAdvisorName', 'TDSamount', 'TotalAmtWtdTax_',
       'SGST_UGST_9','Unnamed_0:1', 'TotalCGST', 'TotalGST', 'TotalIGST', 'TotalSGST_UGST','index:2', 'Unnamed_0:1']]

Cat_data_pb['ItemCategory'].count()
Cat_data_pb['ItemCategory'].value_counts()


cat_data_pb_drop = Cat_data_pb[['CustomerNo_', 'Name2', 'Factorycalendar', 'CITY:1','ClaimNo_', 'ExpiryDate', 'District','GatePassDate', 'GatePassTime', 'Plant:1', 'PlantName1']]

Cat_data_pb.drop(columns = ['CustomerNo_', 'Name2', 'Factorycalendar', 'CITY:1','ClaimNo_', 'ExpiryDate', 'District','GatePassDate', 'GatePassTime', 'Plant:1', 'PlantName1'],inplace = True)




#duration of service time will be invoice date - job card date - 
#there could be chances that if duration is high the customer will not visit the service center again
#Most of the customers have preferred not to have a hard copy
#Technician is not assigned for most of the invoices which are generated

Cat_data_pb.index


Cat_data_pb['Plant'].nunique()

model_revenue = fin_date.groupby(['Plant'])['TotalValue'].sum().reset_index()

#make wise total value
plt.figure(figsize = (20,30))
sns.barplot(data=model_revenue, x="Plant", y="TotalValue")
plt.xticks(rotation=60)
plt.show()


Cat_data_pb.loc[Cat_data_pb['Plant'] == temp1['Plantcode'], 'Plant'] = temp1['code']