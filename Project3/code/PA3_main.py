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

classInputArray = [['https://github.com/IsaacBoyd2/testing/blob/main/Project3/code/preprocessing.py?raw=true','preprocessing.py'],
                   ['https://github.com/IsaacBoyd2/testing/blob/main/Project3/code/loss.py?raw=true', 'loss.py'],
                   ['https://github.com/IsaacBoyd2/testing/blob/main/Project3/code/mlp.py?raw=true', 'mlp.py']]

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

  hiddenArray = [1000]
  eta = 0.0015

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
    modeling.run(len(preProcess.df.iloc[0])-1,hiddenArray,len(preProcess.classes))
    
    
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

    print("\n\n\nTHIS IS THE VALUES: \n", values, "\nACTUAL:", actual)

  #regression
  else:
    modeling = MLP.Model()
    modeling.run(len(preProcess.df.iloc[0])-1,hiddenArray,1)


    type_of_cross_v = 10

    training_size = math.ceil(len(preProcess.df)*(type_of_cross_v-1)/(type_of_cross_v))                          #Split the data on x-cross val

    percentages = []

    random_list = random.sample(range(len(preProcess.df)), len(preProcess.df)) 

    testing_size = len(preProcess.df) - training_size


    testing_list = []
    training_list = []
    testing_list = random_list[0*testing_size:testing_size*(0+1)]

    random_list2 = []

    for k in random_list:
      random_list2.append(k)

    for j in testing_list:
      random_list2.remove(j)

    for l in random_list2:
      training_list.append(l)

    training_df =  preProcess.df.iloc[training_list]
    testing_df_with_labels = preProcess.df.iloc[testing_list]
    testing_df = testing_df_with_labels.iloc[: , :-1]


    #print(training_df[0:10])
    #print(testing_df[0:10])


    #print('df', preProcess.df[0:20])
    #print('df', preProcess.df[-20:])

    #lengthtogo = len(preProcess.df)*0.9

    #lengthtogo = round(lengthtogo)


    #print(modeling.mlp_init)

    
    for q in range(5):
      
      for i in range(len(training_df)):   #len(training_df.iloc[0])

        

        #print('asdlkjfhasdf',len(preProcess.folds[0:8]))


        #print('ehllo',preProcess.df.values[i,0:-1])
        #print(len(preProcess.df.values[i,0:-1].astype('float')))
        #print(preProcess.value)

        modeling.forwardProp(training_df.values[i,0:-1].astype('float'),preProcess.value)
        #print('modeling.output',modeling.output)

        #print('Weights before bp : ',modeling.mlp_init)                  #important

        #print(modeling.output)
        #print(preProcess.df.values[i,-1])
        #print(len(preProcess.df.values[i,-1]))

        
        #print('Training things',training_df.values[i,0:-1])
        #print('The preprocessing actual that we are sending in.',preProcess.df.values[i,-1])


        modeling.Back_Prop(eta,1,training_df.values[i,-1],1)
        #print('Values inside the nodes xis : ',modeling.values)                      #These four important
        #print('Weights connecting the nodes after bp : ',modeling.mlp_init)
        #print('Output of forwards prop : ',modeling.output)


        #print('\n Cycle -------------------------------- \n')

    values = []
    actual = []
    for i in range(len(testing_df)):
      print(f'Goal {i} : ', testing_df_with_labels.values[i,-1])


    for i in range(len(testing_df)):   #len(training_df.iloc[0])

      #print('asdlkjfhasdf',len(preProcess.folds[0:8]))


      #print('ehllo',preProcess.df.values[i,0:-1])
      #print(len(preProcess.df.values[i,0:-1].astype('float')))
      #print(preProcess.value)

      modeling.forwardProp(testing_df.values[i].astype('float'),preProcess.value)
      print(f'\n\nFinal Output {i}: ',modeling.output)

    #print(training_df)

    '''for i in range(len(training_df)):
      #print(modeling.mlp_init)
      modeling.forwardProp(training_df.values[i,0:-1].astype('float'),preProcess.value)
      print(modeling.output)'''
      #print(training_df.iloc[i])

    #for j in range(len(testing_df)):
      #print(modeling.mlp_init)
      #print(modeling.values)
      #modeling.forwardProp(testing_df.values[j,0:-1].astype('float'),preProcess.value)
      #print(testing_df.iloc[j])
      




      #print(modeling.output)




    #print(modeling.mlp_init)
    #for i in range(len(preProcess.df)-lengthtogo-3):

      #print(len(preProcess.df)-lengthtogo)

      
      
      #print(preProcess.df.values[i+len(preProcess.folds[0:8]),0:-1].astype('float'))


      #modeling.forwardProp(preProcess.df.values[i+lengthtogo,0:-1].astype('float'),preProcess.value)

      #print('hhhh',preProcess.df.values[i+lengthtogo,0:-1].astype('float'))
      #print('iiii',preProcess.df.values[i+lengthtogo])
      #print('outputjjj',modeling.output)

      
      #print(modeling.values)
      #print(modeling.output)



      #values.append(modeling.output)
      #print(i)
      #print(len(preProcess.folds[9]))
      #actual.append(preProcess.folds[9][i][-1])

    #print("\n\n\nTHIS IS THE VALUES: \n", values, "\nACTUAL:", actual)

  
 



#calls main
main()

# makes sure that the python classes are taken out after execution
for i in range(len(classInputArray)):
  fileName = classInputArray[i][1]
  os.remove(fileName)
