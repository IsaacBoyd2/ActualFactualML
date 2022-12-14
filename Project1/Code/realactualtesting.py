# -*- coding: utf-8 -*-
"""RealActualTesting

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZwWBjMTopWsRxivryIpaHtswSIaOdOhY
"""

#Final Code

#------------------------Header-------------------------

#Code by: Isaac Boyd, James Lucas 

##Code For: Naive Bayes
##Completed: 9-15-2022
##References: NA

#-----------------------imports-------------------------

import pandas as pd
import numpy as np
import math
import random

#----------------------Functions------------------------

#This function bins data if it is non-discrete.
def binning_func(df,number_of_bins):
  #Go through all of the columns
  for count, columns in enumerate(df.columns[0:len(df.columns)-1]):
    #Split each column into bins
    bin_numbers = np.linspace(df[columns].min(),df[columns].max(),number_of_bins)
    print('Bin Numbers: '+ str(bin_numbers))
    #Go through every value in the column
    for value in range(len(df)):
      closeness_list = []
      #Calculate the closest bin and change the data appropriately
      for the_bins in bin_numbers:
        closeness_list.append(abs(df.iat[value, count]-the_bins))
    print()
    df.iat[value, count] = bin_numbers[closeness_list.index(min(closeness_list))]
  return df

#This function introduces noise to the dataset
def randomizer_func(df):
  #Go through each column
  for counter, columns in enumerate(df.columns[0:len(df.columns)-1]):
    #Get random indexs from each column
    df_random = df.sample(frac=0.10).index
    feature_values = []
    indexer = []
    #Get the information from each index
    for i in df_random:
      feature_values.append(df[f'{columns}'][i])
      indexer.append(i)
    random.shuffle(indexer)
    #Match the original data with shuffled indexes and update dataframe
    for count,j in enumerate(indexer):
      df.iat[count, counter] = feature_values[count]
  return df

#Model/Algorithm
def model(training_df, testing_df,bins):
    #Go through every testing datum
    for lines in range(len(testing_df)):                                       
      row = testing_df.iloc[lines]                              
      C_x = []
      #Split the data into classes (individual dataframes)
      for i in bins:                       
        F_a_c_list = []                                                 
        category_df = training_df[training_df['Class'] == i]
        #For every feature in each data slice count how many similar values there are in the column for every class                         
        for count,j in enumerate(row):                                                                                  
          y = 0
          for k in category_df.iloc[:, count][0:len(category_df)]:              
            if k == j:
              y = y+1  
          numerator = y + (m * p)
          denominator = len(category_df) + m                                                                  
          F_a_c = numerator/denominator
          F_a_c_list.append(F_a_c)
        C_x.append(np.prod(F_a_c_list)*(len(category_df)/len(df)))
      #Output Decision
      results.append(bins[C_x.index(max(C_x))])                                  

#Statistics Calculator
def get_stats(testing_df_with_labels,results,bins,confusionMat):
  #Creates a Confusion Matrix that will be used for precision, recall, and F1 calculations.
  for count,i in enumerate(results):
    #indexes that are used to locate the position in the confusion matrix
    indHorz = 0
    indVert = 0
    #checks for position in the confusion matrix
    for j in range(len(bins)):
      if bins[j] == i:
        indVert = j
      if bins[j] == testing_df_with_labels['Class'].iloc[count]:
        indHorz = j
    #adds an occurance in the confusion matrix
    confusionMat[indHorz, indVert] += 1
  

  if number_of_folds == type_of_cross_v:

    #used to hold all of the precisions and recalls in lists
    precisionList = []
    recallList = []
    F1 = 0

    #loops through all classes to find their respective precisions and recalls
    for i in range(len(bins)):
      falsePos = 0
      falseNeg = 0
      #calculates false positives and false negatives
      for j in range(len(bins)):
        if confusionMat[j, i] != 0 and confusionMat[j, i] != confusionMat[i, i]:
          falsePos += confusionMat[j, i]
        if confusionMat[i, j] != 0 and confusionMat[i, j] != confusionMat[i, i]:
          falseNeg += confusionMat[i, j]
      #makes sure that there is not a divide by 0 error
      if confusionMat[i,i] + falsePos != 0:
        precisionList.append(confusionMat[i,i]/(confusionMat[i,i] + falsePos))
      if confusionMat[i,i] + falseNeg != 0:
        recallList.append(confusionMat[i,i]/(confusionMat[i,i] + falseNeg))
    
    #takes the average of the precisions and recalls of every class for evaluating the F1 statistic
    precisionAvg = np.average(precisionList)
    recallAvg = np.average(recallList)
    if precisionAvg == 0 and recallAvg == 0:
      F1 = 0
    else:
      F1 = 2*((precisionAvg * recallAvg)/(precisionAvg + recallAvg))
    
    #Put in list so that we can take the average of all of our trials for 10-fold cross validation
    average_recall.append(recallAvg)
    average_precision.append(precisionAvg)
    average_f1.append(F1)

#----------------------Main--------------------------

#----Options----

#Introduce noise
shuffle_datasets = False

#1 Breast Cancer ; #2 Glass ; #3 Iris ; #4 soybean ; #5 Voting
dataset = 3

#x Fold Cross validatoin
type_of_cross_v = 10

#----HyperParameters----
number_of_bins = 5
tuning_parameter = 5
thing1 = [3]
thing2 = [1]
thing3 = [1]

final_df = pd.DataFrame(columns=['number_of_bins','m','p','accuracy','f1','recall','precision'])

#loops 30 times for more accurate results
for epochs in range(1):
  results_df = pd.DataFrame(columns=['number_of_bins','m','p','accuracy','f1','recall','precision'])
  for number_of_bins in thing1:
    for m in thing2:
      for p in thing3:

        #Bring in the data
        if dataset == 1: 
          df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/4edb8f698e2425d1e799ae4419663cca285e2e5b/DataProject1/breast-cancer-wisconsin.csv?raw=true')
          #Replace unknown values with the mode per class
          for count, columns in enumerate(df.columns):
            for value in range(len(df)):
              if df.loc[value].at[columns] == '?':
                class_df = df[df['Class'] == df.loc[value].at['Class']]
                df.iat[value, count] = int(class_df[columns].mode())
        elif dataset == 2: 
          df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/59ea51ba9844a15e7afb0d8bd6b92821689fb538/DataProject1/glass.csv?raw=true')
        elif dataset == 3:
          df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/5def86edf4fbfd4680cb5658061188cfd76628d7/DataProject1/iris.csv?raw=true')
        elif dataset == 4:
          df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/eb4cfd5f3920f3038f8d543ceca68080fb5c552f/DataProject1/soybean-small.csv?raw=true')
        elif dataset == 5:
          df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/7010ae4c3d3dc4d9cee1d68c20f20ddb01dfd30d/DataProject1/house-votes-84.csv?raw=true')

        print(df[0:20])

        #----Preprocessing----

        #Get all of the classes
        bins = df['Class'].unique()

        #bin the data if non-discrete
        if type(df.iat[1, 1])==np.float64:
          df = binning_func(df,number_of_bins)

        print(df[0:20])

        #Introduce noise into the dataset
        if shuffle_datasets == True:
          df = randomizer_func(df)

        #Generate indexes and size testing to x-folds
        random_list = random.sample(range(len(df)), len(df)) 
        get_extras = len(df) % type_of_cross_v
        keep = get_extras

        average_accuracy = []
        average_f1 = []
        average_recall = []
        average_precision = []
        confusionMat = np.zeros((len(bins), len(bins)))

        #Execute the algorithm x amount of times
        number_of_folds = 0
        for cross in range(type_of_cross_v):
          number_of_folds += 1
          testing_list = []
          training_list = []
          #Get training and testing indexes (sometimes we do not have an easy sized dataset 
          #so we need to add a couple here and there to the training slices)
          testing_size = len(df) - math.ceil(len(df)*(type_of_cross_v-1)/(type_of_cross_v))
          if get_extras != 0:
            testing_size = testing_size+1
            testing_list = random_list[cross*testing_size:testing_size*(cross+1)]
            get_extras = get_extras-1
          else:
            testing_list = random_list[(cross*testing_size)+keep:testing_size*(cross+1)+keep]

          training_list = random_list.copy()
          for j in testing_list:
            training_list.remove(j)

          #Here we get our training and testing dataframe
          training_df =  df.iloc[training_list]
          testing_df_with_labels = df.iloc[testing_list]

          testing_df = testing_df_with_labels.iloc[: , :-1]
          results = []

          #Call our model and then get statistics from the model
          model(training_df,testing_df,bins)
          get_stats(testing_df_with_labels,results,bins,confusionMat)

        df1 = pd.DataFrame({'number_of_bins':[number_of_bins], 'm':[m],'p':[p], 'accuracy':[np.average(average_accuracy)], 'f1':[np.average(average_f1)], 'recall':[np.average(average_recall)], 'precision':[np.average(average_precision)]})

        results_df =results_df.append(df1)

  final_df = final_df.append(results_df)

df3 = final_df.groupby(['number_of_bins','m','p'], as_index=False)['accuracy','f1','recall','precision'].mean()

output_df = final_df[['f1','precision']].copy()

output_df.to_csv('output')



#----------------------------Graphing---------------------------------#

import matplotlib.pyplot as plt
import pylab
from matplotlib import rc, rcParams
pylab.rcParams['xtick.major.pad']='25'
X = df3['m']
Y = df3['p']
Z = df3['f1']

plt.rcParams.update({'font.size': 30})

pylab.rcParams['xtick.major.pad']='40'
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(X, Y, Z,  cmap='viridis', linewidth=0.5);
fig.set_size_inches(20, 20)
ax.view_init(15,15)
ax.set_xlabel('M',labelpad = 60,fontweight='bold')
ax.set_ylabel('P',labelpad = 60,fontweight='bold')
ax.set_zlabel('F1 Score',labelpad = 80,fontweight='bold')

print('---------------\n'+'|TP:'+str(int(confusionMat[0,0]))+'|'+'FP:'+str(int(confusionMat[0,1]))+' |\n|-------------|'+'\n|FN:'+str(int(confusionMat[1,0]))+'  |'+'TN:'+str(int(confusionMat[1,1]))+'|'+'\n---------------')