# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 10:49:54 2021

@author: Komali Srinivas
"""

a = ['I','am','a','good','data','scientist','this','is','my','assumption']
longetword = ''
for i in a:
    if len(i)<=5:
        print(i)
        if len(i)>len(longetword):
            longetword = i
print('longest word with length lessthan 5 char is:',longetword)

#pip install pep8



