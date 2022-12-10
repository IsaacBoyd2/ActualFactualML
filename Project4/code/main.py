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
                   ['https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project4/code/mlp.py?raw=true', 'mlp.py']]

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

  hiddenArray = [10]

  particleNumber = 30

  #------run-properties--------#

  algType = 3

  #----------------------------#

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
      

      #Makes the testing and training sets
      preProcess.oneHot()

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
      
      #------------GA------------
      if algType == 1:
        pass
      
      #------------GE------------
      elif algType == 2:
        pass
      
      #----------PSO------------
      elif algType == 3:
        
        #used to hold the particles
        models = []

        #makes the defined number of particles
        for i in range(particleNumber):
          model = MLP.Model()
          model.run(len(preProcess.df.iloc[0]-1), hiddenArray, len(preProcess.classes))
          models.append(model) 

        topModel = MLP.Model()
        modeling = topModel.algPSO(models, 0, training_df, testing_df, testing_df_with_labels)
      
      actual = []
      values = []
      uniques = training_df['Class'].unique()
      #Used to test the model with the testing data.
      for i in range(len(testing_df)):
        modeling.forwardProp(testing_df.values[i].astype('float'),preProcess.value)
        actual.append(testing_df_with_labels.values[i,-1])
        values.append(uniques[modeling.values[-1].index(max(modeling.values[-1]))])

    lossValues = lss.Loss()
    lossValues.calculate(uniques, values, actual)
    print("\n\nF1-score after testing: ", lossValues.F1)

  #regression
  else:

    #Define what type of cross fold validation we will be doing
    type_of_cross_v = 10

    #Get the size of the training
    training_size = math.ceil(len(preProcess.df)*(type_of_cross_v-1)/(type_of_cross_v))                          #Split the data on x-cross val
    percentages = []

    #Get the size of the testing
    random_list = random.sample(range(len(preProcess.df)), len(preProcess.df)) 
    testing_size = len(preProcess.df) - training_size

    totalvalues = []
    totalactual = []

    for run_num in range(1):   #Just for 1
      #Makes the testing and training sets
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
      
      #------------GA------------
      if algType == 1:
        pass
      
      #------------GE------------
      elif algType == 2:
        pass
      
      #----------PSO------------
      elif algType == 3:
        foldCounter = 0
        while foldCounter < 10:
          #used to hold the particles
          models = []

          #makes the defined number of particles
          for i in range(particleNumber):
            model = MLP.Model()
            model.run(len(preProcess.df.iloc[0]-1), hiddenArray, 1)
            models.append(model) 

          topModel = MLP.Model()
          modeling = topModel.algPSO(models, 1, training_df, testing_df, testing_df_with_labels)
        
          #Used to test the model with the testing data.
          for i in range(len(testing_df)):
            modeling.forwardProp(testing_df.values[i].astype('float'),preProcess.value)
            totalactual.append(testing_df_with_labels.values[i,-1])
            totalvalues.append(modeling.output)

          foldCounter += 1

    lossValues = lss.Loss()
    lossValues.calculateReg(totalvalues, totalactual)
    print("\n\nMSE after testing: ", lossValues.mse)
    

#calls main
main()

# makes sure that the python classes are taken out after execution
for i in range(len(classInputArray)):
  fileName = classInputArray[i][1]
  os.remove(fileName)
