# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 02:26:37 2021

@author: karti

All inactive code is there for experimentation of different methods
to help build the model.
"""

#import numpy as np
import pandas as pd
import xgboost as xgb
import re
from sklearn import model_selection,metrics

#from grad_desc import grad_descent
#from sigmoid import sigmoid
#from CostFunction import costfunction
#from sklearn import preprocessing

def getNumbers(str): 
    array = re.findall(r'[0-9]', str) 
    return array


#Load data
train_dat = pd.read_csv("train.csv")

#Reformat Cabin, Tickets, Title, Fare, Age.
train_dat['Cabin_Class'] = train_dat['Cabin'].str[0]
train_dat['Cabin_Class'].replace({'A':int(1),'B':int(2),'C':int(3),'D':int(4),'E':int(5),'F':int(6),'G':int(7),'T':int(8)},inplace=True)
train_dat['Cabin_Class'] = train_dat['Cabin_Class'].fillna(0)



#Fill in missing data by using existing trends in the existing data set
#to extrapolate a likely replacement for missing values. This greatly
#helped increase the accuracy of the model.

for i in range(train_dat.shape[0]):
    kl = getNumbers(train_dat.loc[i,'Ticket'])
    if kl == []:
        train_dat.loc[i,'Ticket'] = 0
    else:
        train_dat.loc[i,'Ticket'] = int(kl[0])

train_dat.loc[:,'Ticket'] = train_dat.loc[:,'Ticket'].astype(int)



train_dat["Title"] = train_dat["Name"].str.extract('([A-Za-z]+\.)',expand=False)

others = ['Capt.','Don.','Jonkheer.','Rev.','Dona.']

for other in others:
    train_dat.loc[train_dat['Title'] == other,'Title'] = 'Other'

safes = ['Countess.','Lady.','Mlle.','Mme.','Ms.','Sir.']

for safe in safes:
    train_dat.loc[train_dat['Title'] == safe,'Title'] = 'Safe'

tittles1 = list(set(train_dat["Title"]))

for i in range(len(tittles1)):
    train_dat.loc[train_dat['Title'] == tittles1[i],'Title'] = i

tr = train_dat[['Age','Title']].groupby(['Title']).mean()

for i in range(tr.shape[0]):
    train_dat.loc[train_dat['Title']==i,'Age'] = tr.loc[:,'Age'][i]

train_dat.loc[:,'Title'] = train_dat.loc[:,'Title'].astype(int)

train_dat.loc[train_dat['Age']<=14, 'Age'] = 0
train_dat.loc[(train_dat['Age']>14) & (train_dat['Age']<=27), 'Age'] = 1
train_dat.loc[(train_dat['Age']>27) & (train_dat['Age']<=41), 'Age'] = 2
train_dat.loc[(train_dat['Age']>41) & (train_dat['Age']<=54), 'Age'] = 3
train_dat.loc[(train_dat['Age']>54) & (train_dat['Age']<=67), 'Age'] = 4
train_dat.loc[(train_dat['Age']>67) & (train_dat['Age']<=80), 'Age'] = 5

train_dat.loc[train_dat['Fare']<=7.7, 'Fare'] = 0
train_dat.loc[(train_dat['Fare']>7.7) & (train_dat['Fare']<=7.9), 'Fare'] = 1
train_dat.loc[(train_dat['Fare']>7.9) & (train_dat['Fare']<=8.7), 'Fare'] = 2
train_dat.loc[(train_dat['Fare']>8.7) & (train_dat['Fare']<=13), 'Fare'] = 3
train_dat.loc[(train_dat['Fare']>13) & (train_dat['Fare']<=16.7), 'Fare'] = 4
train_dat.loc[(train_dat['Fare']>16.7) & (train_dat['Fare']<=26), 'Fare'] = 5
train_dat.loc[(train_dat['Fare']>26) & (train_dat['Fare']<=35.1), 'Fare'] = 6
train_dat.loc[(train_dat['Fare']>35.1) & (train_dat['Fare']<=73.5), 'Fare'] = 7
train_dat.loc[(train_dat['Fare']>73.5) & (train_dat['Fare']<=523), 'Fare'] = 8




#Remove redundant columns.
headers = list(train_dat.columns.values)
redundants = ['PassengerId','Name','Cabin','Ticket']
headers = list(set(headers) - set(redundants))
newtrain = train_dat.loc[:,headers]

#Drop rows with Nan.
newtrain = newtrain.dropna(axis=0,how='any')


#Create outcome vector.
outcome = newtrain.loc[:,'Survived']
newtrain = newtrain.drop(['Survived'],axis=1)
outcome = outcome.reset_index(drop=True)



#Reformat the Embarked and Sex columns.
newtrain = newtrain.reset_index(drop=True)
newtrain['Embarked'].replace({'C':1,'Q':2,'S':3},inplace=True)
newtrain['Sex'].replace({'male':0,'female':1},inplace=True)
#newtrain.insert(0,'Constant',np.ones(newtrain.shape[0]),True)

#fare_mean = newtrain['Fare'].mean()
#fare_std = newtrain['Fare'].std()
#age_mean = newtrain['Age'].mean()
#age_std = newtrain['Age'].std()


#Making Family Column
#comb = []
#for i in range(newtrain.shape[0]):
#    if newtrain['SibSp'][i]>0 or newtrain['Parch'][i]>0:
#        comb.append(1)
#    else:
#        comb.append(0)
#newtrain.insert(3,'Family',np.array(comb))

#Regularisation
#training_set = newtrain[:427][:]
#newtrain['Fare'] = (newtrain['Fare']-fare_mean)/(fare_std)
#newtrain['Age'] = (newtrain['Age']-age_mean)/(age_std)





#Load test data.
test_dat = pd.read_csv("test.csv")
test_data = test_dat
#Reformat Cabin and Tickets.
test_data['Cabin_Class'] = test_data['Cabin'].str[0]
test_data['Cabin_Class'].replace({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8},inplace=True)
test_data['Cabin_Class'] = test_data['Cabin_Class'].fillna(0)

for i in range(test_data.shape[0]):
    kl = getNumbers(test_data.loc[i,'Ticket'])
    if kl == []:
        test_data.loc[i,'Ticket'] = 0
    else:
        test_data.loc[i,'Ticket'] = int(kl[0])
    
test_data.loc[:,'Ticket'] = test_data.loc[:,'Ticket'].astype(int)
    


test_data["Title"] = test_data["Name"].str.extract('([A-Za-z]+\.)',expand=False)

others = ['Capt.','Don.','Jonkheer.','Rev.','Dona.']

for other in others:
    test_data.loc[test_data['Title'] == other,'Title'] = 'Other'

for safe in safes:
    test_data.loc[test_data['Title'] == safe,'Title'] = 'Safe'

#tittles2 = list(set(test_data["Title"]))

for i in range(len(tittles1)):
    test_data.loc[test_data['Title'] == tittles1[i],'Title'] = i

for i in range(tr.shape[0]):
    test_data.loc[test_data['Title']==i,'Age'] = tr.loc[:,'Age'][i]

test_data.loc[:,'Title'] = test_data.loc[:,'Title'].astype(int)

test_data.loc[test_data['Age']<=14, 'Age'] = 0
test_data.loc[(test_data['Age']>14) & (test_data['Age']<=27), 'Age'] = 1
test_data.loc[(test_data['Age']>27) & (test_data['Age']<=41), 'Age'] = 2
test_data.loc[(test_data['Age']>41) & (test_data['Age']<=54), 'Age'] = 3
test_data.loc[(test_data['Age']>54) & (test_data['Age']<=67), 'Age'] = 4
test_data.loc[(test_data['Age']>67) & (test_data['Age']<=80), 'Age'] = 5
#test_data['Age'] = test_data['Age'].astype(int)

test_data.loc[test_data['Fare']<=7.7, 'Fare'] = 0
test_data.loc[(test_data['Fare']>7.7) & (test_data['Fare']<=7.9), 'Fare'] = 1
test_data.loc[(test_data['Fare']>7.9) & (test_data['Fare']<=8.7), 'Fare'] = 2
test_data.loc[(test_data['Fare']>8.7) & (test_data['Fare']<=13), 'Fare'] = 3
test_data.loc[(test_data['Fare']>13) & (test_data['Fare']<=16.7), 'Fare'] = 4
test_data.loc[(test_data['Fare']>16.7) & (test_data['Fare']<=26), 'Fare'] = 5
test_data.loc[(test_data['Fare']>26) & (test_data['Fare']<=35.1), 'Fare'] = 6
test_data.loc[(test_data['Fare']>35.1) & (test_data['Fare']<=73.5), 'Fare'] = 7
test_data.loc[(test_data['Fare']>73.5) & (test_data['Fare']<=523), 'Fare'] = 8
#test_data['Fare'] = test_data['Fare'].astype(int)


    

#Remove redundant columns.
test_headers = list(test_data.columns.values)
test_headers = list(set(test_headers) - set(redundants))
test_data = test_data.loc[:,test_headers]


#Reformat the Embarked and Sex columns.
test_data['Embarked'].replace({'C':1,'Q':2,'S':3},inplace=True)
test_data['Sex'].replace({'male':0,'female':1},inplace=True)
#test_data.insert(0,'Constant',np.ones(test_data.shape[0]),True)
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
test_data = test_data.fillna(0)
#test_data[list(newtrain.columns.values)]

#Regularisation
#test_data['Fare'] = (test_data['Fare']-fare_mean)/(fare_std)
#test_data['Age'] = (test_data['Age']-age_mean)/(age_std)


#Create Family Columns
#fam = []
#for i in range(test_data.shape[0]):
#    if test_data['SibSp'][i]>0 or test_data['Parch'][i]>0:
#        fam.append(1)
#    else:
#        fam.append(0)
    
#test_data.insert(3,'Family',np.array(fam))
#test_data[list(newtrain.columns.values)]





#Training the model and converting to DMatrix.

#dtrain = xgb.DMatrix(newtrain,label = outcome)
#dtest = xgb.DMatrix(test_data)

#param = {'objective': 'binary:logistic', 'eval_metric':'error', 'eta':0.1,
#         'max_depth':7}
#num_round = 30

#bst = xgb.train(param, dtrain, num_round)
#bst.save_model('trial.model')


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(newtrain, outcome,
                                                                    test_size=0.3,stratify=outcome,random_state=0)
xgboost_model = xgb.XGBClassifier(objective='binary:logistic',learning_rate=0.1)
eval_set = [(train_x,train_y),(valid_x,valid_y)]

xgboost_model.fit(train_x,train_y,eval_metric=['error','logloss','auc'],eval_set=eval_set,verbose=True)

xgboost_model.score(train_x,train_y)
pred_y = xgboost_model.predict(valid_x)
metrics.accuracy_score(valid_y,pred_y)

pred_test = xgboost_model.predict(test_data)

sub = pd.DataFrame({'PassengerId':test_dat['PassengerId'],'Survived':pred_test})

#Prediction on test set.




#trans = bst.predict(dtest)


#trans = trans>=0.5


#for i in range(trans.shape[0]):
#    trans[i] = int(trans[i])

#transwer = trans.astype(int)

#answer1 = pd.DataFrame(transwer)
#answer1.insert(0,'PassengerId',np.arange(892,1310),True)
#answer1.columns = ['PassengerId','Survived']


#from matplotlib import pyplot

#pyplot.scatter(newtrain[''], outcome)




