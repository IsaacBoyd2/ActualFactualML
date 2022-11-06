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

  hiddenArray = [4,3]
  eta = .001

  #----------------------------#

  #runs preProcessing
  preProcess = pp.Preprocessing()
  preProcess.process()
  preProcess.normalize()
  preProcess.fold()
  data = preProcess.folds

  #classification
  if preProcess.value == 0:
    preProcess.oneHot()
    modeling = MLP.Model()
    modeling.run(len(preProcess.df.iloc[0]),hiddenArray,len(preProcess.classes))
    
    
    for i in range(len(preProcess.folds[0:8])): 
      modeling.forwardProp(preProcess.df.values[i,0:-1].astype('float'),preProcess.value)
      modeling.Back_Prop(eta,0,[preProcess.oneHotDict,preProcess.df.values[i,-1]],len(preProcess.classes))
      
    values = []
    actual = []
    for i in range(len(preProcess.folds[9])):
      modeling.forwardProp(preProcess.df.values[i+len(preProcess.folds[0:8]),0:-1].astype('float'),preProcess.value)
      values.append(modeling.values[-1].index(max(modeling.values[-1])))
      actual.append(preProcess.folds[9][i][-1])

    for i in range(len(values)):
      print(preProcess.classes)
      values[i] = preProcess.classes[values[i]]

    loss = lss.Loss()
    loss.calculate(preProcess.classes, values, actual)
    print("F1: ", loss.F1)

  #regression
  else:
    modeling = MLP.Model()
    modeling.run(len(preProcess.df.iloc[0]),hiddenArray,1)

    for q in range(30):
      for i in range(len(preProcess.folds[0:8])):
        modeling.forwardProp(preProcess.df.values[i,0:-1].astype('float'),preProcess.value)
        modeling.Back_Prop(eta,1,preProcess.df.values[i,-1],1)

    values = []
    actual = []
    for i in range(len(preProcess.folds[9])):
      modeling.forwardProp(preProcess.df.values[i+len(preProcess.folds[0:8]),0:-1].astype('float'),preProcess.value)
      values.append(modeling.output)
      actual.append(preProcess.folds[9][i][-1])

    loss = lss.Loss()
    loss.calculateReg(values, actual)
    print("MSE: ", loss.mse, "\nMAE: ", loss.mae)

  
 



#calls main
main()

# makes sure that the python classes are taken out after execution
for i in range(len(classInputArray)):
  fileName = classInputArray[i][1]
  os.remove(fileName)
