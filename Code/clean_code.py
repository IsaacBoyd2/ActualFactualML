#import Pandas and Numpy

import pandas as pd
import numpy as np
import math
import random

#--------------------------------------


#Hyper Parameters

dev = 16 #standard deviation control                                  #May need to adjust for soybean
balance = True #Set to true to balance datasets
dataset = 4  #Set to dataset of choice                                #1 Breast Cancer ; #2 Glass ; #3 Iris ; #4 soybean ; #5 Voting

#--------------------------------------
#Bring in the data

if dataset == 1:
  df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/4edb8f698e2425d1e799ae4419663cca285e2e5b/DataProject1/breast-cancer-wisconsin.csv?raw=true')
  df = df.replace('?', 0)
  df = df.apply(pd.to_numeric)
elif dataset == 2:
  df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/59ea51ba9844a15e7afb0d8bd6b92821689fb538/DataProject1/glass.csv?raw=true')
elif dataset == 3:
  df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/5def86edf4fbfd4680cb5658061188cfd76628d7/DataProject1/iris.csv?raw=true')
elif dataset == 4:
  df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/eb4cfd5f3920f3038f8d543ceca68080fb5c552f/DataProject1/soybean-small.csv?raw=true')
#elif dataset == 5:


#--------------------------------------
#Preprocessing


bins = df['Class'].unique()                                #Get all of the classes

training_size = math.ceil(len(df)*4/5)                          #Split the data 80/20 by index

percentages = []


for sdlfj in range(177):

  random_list = random.sample(range(len(df)), len(df)) 


  #Create a list of training and testing data
  training_list = random_list[0:training_size]
  testing_list = random_list[training_size:len(df)]

  #Create a testing and training dataframe from the lists
  training_df =  df.iloc[:, 0:len(df.columns)]
  testing_df_with_labels = df.iloc[testing_list]
  testing_df = testing_df_with_labels.iloc[: , :-1]

  #category_df = training_df[training_df['Glass Type'] == 1]
  #print(category_df.loc[random.randint(0, len(category_df)-1)])

  def balancing(training_df):

    max_list = []

    for i in bins:
      category_df = training_df[training_df['Class'] == i]
      max_list.append(len(category_df))
    
    for i in bins:
      category_df = training_df[training_df['Class'] == i]
      while len(category_df) < max(max_list):
        #df2 = df2.append(df1.iloc[x])
        training_df = training_df.append(category_df.iloc[random.randint(0, len(category_df)-1)])
        category_df = training_df[training_df['Class'] == i]
        #print(len(category_df))

    return training_df

  if balance == True:
    training_df = balancing(training_df)

  #--------------------------------------


  #Model/Algorithm

  '''
  Breif Description of model:
  Model takes in a training and testing dataframe. Each entry into the training dataframe is
  an unknown entry with several attributes. These attributes are compared to those in the training
  set and a estimation is made based on a range as to which class shares the most attributes with
  the input data.
  '''

  results = []

  def model(training_df, testing_df):

    for lines in range(len(testing_df)):                                         #For every testing datum
      row = testing_df.iloc[lines]                              
      C_x = []
      for i in bins:                                                             #For every class (so in other words every testing input will be compared to every class)
        F_a_c_list = []                                                 
        category_df = training_df[training_df['Class'] == i]  #This splits the data into class spesific dataframes       
        for count,j in enumerate(row):                                                                                  
          y = 0
          for k in category_df.iloc[:, count][0:len(category_df)]:               #For every attribute in the inputs
            #print(k)
            sd = np.std(category_df.iloc[:, count][0:len(category_df)])          #Compare every attribute to all other attributes in the class
            if k < (j+(sd/dev)) and k>(j-(sd/dev)):                                #If the attribute is close to another attribute (withing a fration of an standard deviation) add one
              y = y + 1
            
          numerator = y + 1
          denominator = len(category_df)+len(testing_df.columns)-1               #The following computes for the similarity based on the algotrim
                                                                                #C(x) = Q(C = ci) × d∏j=1 F (Aj = ak, C = ci)
          F_a_c = numerator/denominator
          F_a_c_list.append(F_a_c)

        C_x.append(np.prod(F_a_c_list)*(len(category_df)/len(df)))

      results.append(bins[C_x.index(max(C_x))])                                  #Append the results for future comparisons

  model(training_df,testing_df)


  #--------------------------------------

  #Analysis (WIP)

  correct = 0
  for count,i in enumerate(results):
    if testing_df_with_labels['Class'].iloc[count] == i:
      correct += 1

  accuracy = correct/len(results)
  print(accuracy)

  print(results)
  print(testing_df_with_labels['Class'])
  percentages.append(accuracy)

