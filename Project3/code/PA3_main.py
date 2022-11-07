from posixpath import join

#------------------------Header-------------------------

#Code by: Isaac Boyd, James Lucas 

##Code For: Neural Net
##Completed: 11-1-2022
##References: NA

#-----------------------imports-------------------------

import pandas as pd
import requests
import os
import random
import math

#----Python Classes import----

classInputArray = [['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/code/preprocessing.py?raw=true','preprocessing.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/code/loss.py?raw=true', 'loss.py'],
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/code/mlp.py?raw=true', 'mlp.py']]

#writes our code from github to our current local file
for i in range(len(classInputArray)):
  with open(classInputArray[i][1], 'w') as f:
    r = requests.get(classInputArray[i][0])
    f.write(r.text)

#imports our classes
import preprocessing as pp
import loss as lss
import mlp as MLP

#------------------------Main--------------------------

def main():

  #-----Hyper_parameters-------#

  hiddenArray = [20]
  
  eta = 0.0015

  superscore = 0

  #----------------------------#

  flag = 0
  if len(hiddenArray) == 0:
    flag = 1

  #runs preProcessing
  preProcess = pp.Preprocessing()
  preProcess.process()
  preProcess.normalize()
  preProcess.fold()
  data = preProcess.folds

  #classification
  if preProcess.value == 0:


    #Define what type of cross fold validation we will be doing
    type_of_cross_v = 10


    #Get the size of the training
    training_size = math.ceil(len(preProcess.df)*(type_of_cross_v-1)/(type_of_cross_v))                          #Split the data on x-cross val
    percentages = []

    #Get the size of the testing
    random_list = random.sample(range(len(preProcess.df)), len(preProcess.df)) 
    testing_size = len(preProcess.df) - training_size

    for run_num in range(1):   #Just for 1
      

      #First run our model to initilize the weights
      preProcess.oneHot()
      modeling = MLP.Model()
      
      modeling.run(len(preProcess.df.iloc[0])-1,hiddenArray,len(preProcess.classes),flag)

      testing_list = []
      training_list = []

      testing_list = random_list[run_num*testing_size:testing_size*(run_num+1)]

      random_list2 = []

      for k in random_list:
        random_list2.append(k)

      for j in testing_list:
        random_list2.remove(j)

      for l in random_list2:
        training_list.append(l)

      #here are dataframes that we can acually use 
      training_df =  preProcess.df.iloc[training_list]
      testing_df_with_labels = preProcess.df.iloc[testing_list]
      testing_df = testing_df_with_labels.iloc[: , :-1]
      
      
      
      for q in range(60):
        
        for i in range(len(training_df)):   
          modeling.forwardProp(training_df.values[i,0:-1].astype('float'),preProcess.value)
          modeling.Back_Prop(eta,0,[preProcess.oneHotDict,training_df.values[i,-1]],len(preProcess.classes))
          
        values = []
        actual = []

      uniques = training_df['Class'].unique()
      
      for i in range(len(testing_df)):
        modeling.forwardProp(testing_df.values[i].astype('float'),preProcess.value)
        actual.append(testing_df_with_labels.values[i,-1])
        values.append(uniques[modeling.values[-1].index(max(modeling.values[-1]))])

        if testing_df_with_labels.values[i,-1] == uniques[modeling.values[-1].index(max(modeling.values[-1]))]:
          superscore += 1

    print(superscore)
    print(preProcess.df)


    print(superscore/len(preProcess.df))

    lossValues = lss.Loss()
    lossValues.calculate(uniques, values, actual)

  #regression
  else:


    #First run our model to initilize the weights
    modeling = MLP.Model()
    modeling.run(len(preProcess.df.iloc[0])-1,hiddenArray,1,flag)


    #Define what type of cross fold validation we will be doing
    type_of_cross_v = 10

    #Get the size of the training
    training_size = math.ceil(len(preProcess.df)*(type_of_cross_v-1)/(type_of_cross_v))                          #Split the data on x-cross val
    percentages = []

    #Get the size of the testing
    random_list = random.sample(range(len(preProcess.df)), len(preProcess.df)) 
    testing_size = len(preProcess.df) - training_size


    testing_list = []
    training_list = []
    #This is hardcoded but will need to be changed to fit crossv
    testing_list = random_list[0*testing_size:testing_size*(0+1)]

    random_list2 = []

    for k in random_list:
      random_list2.append(k)

    for j in testing_list:
      random_list2.remove(j)

    for l in random_list2:
      training_list.append(l)

    #here are dataframes that we can acually use 
    training_df =  preProcess.df.iloc[training_list]
    testing_df_with_labels = preProcess.df.iloc[testing_list]
    testing_df = testing_df_with_labels.iloc[: , :-1]

    
    for q in range(30):
      
      for i in range(len(training_df)):   

        modeling.forwardProp(training_df.values[i,0:-1].astype('float'),preProcess.value)

        modeling.Back_Prop(eta,1,training_df.values[i,-1],1)

    values = []
    actual = []
    for i in range(len(testing_df)):
      actual.append(testing_df_with_labels.values[i,-1])


    for i in range(len(testing_df)):
      modeling.forwardProp(testing_df.values[i].astype('float'),preProcess.value)
      print(f'Goal {i} : ', testing_df_with_labels.values[i,-1])
      print(f'Final Output {i}: ',modeling.output)
      print('\n\n')
      values.append(modeling.output)

    lossValues = lss.Loss()
    lossValues.calculateReg(values, actual)
    

#calls main
main()

# makes sure that the python classes are taken out after execution
for i in range(len(classInputArray)):
  fileName = classInputArray[i][1]
  os.remove(fileName)
