#------------------------Header-------------------------

#Code by: Isaac Boyd, James Lucas 

##Code For: Neural Net
##Completed: 10-9-2022
##References: NA

#-----------------------imports-------------------------

import pandas as pd
import numpy as np
from scipy import stats as st
import requests
import os
import math
import random as random
import sys

with open('loss.py', 'w') as f:
  r = requests.get('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/code/loss.py?raw=true')
  f.write(r.text)

import loss as lss
#----------------------classes-------------------------

class Model:

  def __init__(self):
    self.df = pd.DataFrame
    self.predictions = []
    self.labels = []
    self.mlp_init = []
    self.values = []
    
  def run(self, input_size, hidden_sizes, output_size):  
    #Initilize the network to have random weights between 0 and 1.

    mlp_init = []   #The network

    #Input
    hidden_nodes = []
    for i in range(input_size):
      hidden_node = []
      for i in range(hidden_sizes[0]):
        hidden_node.append(random.random())
      hidden_nodes.append(hidden_node)

    mlp_init.append(hidden_nodes)
    

    #Init number of weights between each hidden layer
    for sizes in range(len(hidden_sizes)-1):
      hidden_nodes = []
      for i in range(hidden_sizes[0+sizes]):
        hidden_node = []
        for j in range(hidden_sizes[1+sizes]):
          hidden_node.append(random.random())
        hidden_nodes.append(hidden_node)

      mlp_init.append(hidden_nodes)

    #output layer
    output_nodes = []
    for i in range(hidden_sizes[-1]):
      output_node = []
      for i in range(output_size):
        output_node.append(random.random())
      output_nodes.append(output_node)
    
    
    mlp_init.append(output_nodes)

    #print(mlp_init)

    

    self.mlp_init = mlp_init

    #print(self.mlp_init)

    '''

    mlp_init is set up in the following fashion.

    input layer ->    input_nodes[weight_going_out,weight_going_out,weight_going_out ... weight_going_out]
    hiden layer 1 ->  hidden_nodes[weight_going_out,weight_going_out,weight_going_out ... weight_going_out]
    ...
    hideen layer x -> hidden_nodes[weight_going_out,weight_going_out,weight_going_out ... weight_going_out]
    output layer ->   ouput_nodes[weight_going_out,weight_going_out,weight_going_out ... weight_going_out]

    '''
    
  def forwardProp(self,input,classNumber):      #potentially need to do something for just the input layers
    values = [[]]
    values[0] = input
    #loops through each layer.
    for i in range(len(self.mlp_init)):
      layer_outputs = []
      #loops through each node      
      if i != len(self.mlp_init)-1: #As long as we are not in the last layer
        for j in range(len(self.mlp_init[i+1])):  #This grabs the length of the next layer
          l = []
          for k in range(len(values[i])):   #for every xi
            l.append(float(values[i][k])*float(self.mlp_init[i][k][j]))  #do xiwi
          summation = sum(l) #Sum of all xiwis
          if summation > 50 or summation < -50:
            sigmoid = 1
          else:
            sigmoid = float(1/(1+math.e**(-summation)))    #sigmoid function
          layer_outputs.append(sigmoid) #append for each input

        values.append(layer_outputs) #append all the outputs. (this will be what is "inside" of each node)

      #output layer
      elif classNumber == 1:
        for i in range(1):
          l = []
          for k in range(len(values[-1])):   #for every xi
            l.append(float(values[-1][k])*float(self.mlp_init[-2][k][i]))  #do xiwi
          summation = sum(l)
        
        output = summation

        self.output = output

      #Decision Circuit
      else:
        layer_outputs = []
        for i in range(len(self.mlp_init[-1][0])):
          l = []
          for k in range(len(values[-1])):   #for every xi
            l.append(float(values[-1][k])*float(self.mlp_init[-1][k][i]))  #do xiwi
          summation = sum(l) #Sum of all xiwis
          if summation > 50 or summation < -50:
            sigmoid = 1
          else:
            sigmoid = float(1/(1+math.e**(-summation)))    #sigmoid function
          layer_outputs.append(sigmoid) #append for each input

        values.append(layer_outputs) #append all the outputs. (this will be what is "inside" of each node)
      
      if classNumber == 0:

        sum_of_soft = []
        for i in values[-1]:
          if i > 50 or i < -50:
            softmax1 = 1
          else: 
            softmax1 = float(math.e**i)
          sum_of_soft.append(softmax1)
          
        the_sum_of_soft = sum(sum_of_soft)

        output_values = []

        for i in values[-1]:
          if i > 50 or i < -50:
            softmax2 = 1
          else:
            softmax2 = float((math.e**i)/the_sum_of_soft)
          output_values.append(softmax2)

        values[-1] = output_values

      
    self.values = values


  def algGA(self):
    pass

  def algGE(self):
    pass

  def algPSO(self, particles, dataType, training_df, testing_df, testing_df_with_labels):
    #----Hyper-Parameters----

    c_1 = 1.1
    c_2 = 1.496

    #----Variables----
    parts = particles
    uniques = training_df['Class'].unique()
    r_1 = 0
    r_2 = 0
    loops = 400
    pb = []

    #used to count all of the weights in the model objects (used for velocity)
    totalEdges = 0
    for i in range(len(parts[0].mlp_init)-1):
      totalEdges += len(parts[0].mlp_init[i]) * len(parts[0].mlp_init[i+1])
    totalEdges += len(uniques) * len(parts[0].mlp_init[-1])

    #make a velocity array that will hold 0 velocities for all weights
    v = np.zeros((len(parts), totalEdges))
    gb = [0, np.zeros(totalEdges)]
    
    #classification
    if dataType == 0:
      for i in range(loops):
        print("\n\nLOOP ", i)
        for j in range(len(parts)):
          #used to hold the correct and the guessed classes
          values = []
          actual = []

          #the first particle
          if i == 0 and j == 0:
            for k in range(len(training_df)):
              parts[j].forwardProp(training_df.iloc[k,0:-1].values.astype('float'), dataType)
              actual.append(training_df.values[k,-1])
              values.append(uniques[parts[j].values[-1].index(max(parts[j].values[-1]))])

            lossValues = lss.Loss()
            lossValues.calculate(uniques, values, actual)
            
            #used to hold all values of the weights themselves
            temp = []
            for k in range(len(parts[j].mlp_init)):
              for l in range(len(parts[j].mlp_init[k])):
                for m in range(len(parts[j].mlp_init[k][l])):
                  temp.append(parts[j].mlp_init[k][l][m])

            #sets the global best and the personal best because it is the first loop
            gb = [lossValues.F1, temp]
            pb.append([lossValues.F1, temp])
            print(j, lossValues.F1)

          #the first loop through the particles
          elif i == 0:
            for k in range(len(training_df)):
              parts[j].forwardProp(training_df.iloc[k,0:-1].values.astype('float'), dataType)
              actual.append(training_df.values[k,-1])
              values.append(uniques[parts[j].values[-1].index(max(parts[j].values[-1]))])

            lossValues = lss.Loss()
            lossValues.calculate(uniques, values, actual)
            print(j, lossValues.F1)

            #used to hold all values of the weights themselves
            temp = []
            for k in range(len(parts[j].mlp_init)):
              for l in range(len(parts[j].mlp_init[k])):
                for m in range(len(parts[j].mlp_init[k][l])):
                  temp.append(parts[j].mlp_init[k][l][m])

            if gb[0] < lossValues.F1:
              gb = [lossValues.F1, temp]
            
            pb.append([lossValues.F1, temp])

          #second or more loops through the particles
          else:
            #runs the model with the given training set
            for k in range(len(training_df)):
              parts[j].forwardProp(training_df.iloc[k,0:-1].values.astype('float'), dataType)
              actual.append(training_df.values[k,-1])
              values.append(uniques[parts[j].values[-1].index(max(parts[j].values[-1]))])

            #calculates loss on that model
            lossValues = lss.Loss()
            lossValues.calculate(uniques, values, actual)
            print(j, lossValues.F1)

            #check for personal best F1 score
            if pb[j][0] < lossValues.F1:
              #used to hold all values of the weights themselves
              temp = []
              for k in range(len(parts[j].mlp_init)):
                for l in range(len(parts[j].mlp_init[k])):
                  for m in range(len(parts[j].mlp_init[k][l])):
                    temp.append(parts[j].mlp_init[k][l][m])

              pb[j] = [lossValues.F1, temp]

            #check for global best F1 score
            if gb[0] < lossValues.F1:
              gb = [lossValues.F1,temp]
              print("GLOBAL BEST: ", i, gb[0])
          
            


        for j in range(len(parts)):
          counter = 0
          #update velocity and position
          for k in range(len(parts[j].mlp_init)):
            for l in range(len(parts[j].mlp_init[k])):
              for m in range(len(parts[j].mlp_init[k][l])):
                r_1 = random.uniform(0,1)
                r_2 = random.uniform(0,1)
                v[j][counter] = (.000001*v[j][counter]) + c_1*r_1*(pb[j][1][counter] - parts[j].mlp_init[k][l][m]) + c_2*r_2*(gb[1][counter] - parts[j].mlp_init[k][l][m])
                parts[j].mlp_init[k][l][m] = parts[j].mlp_init[k][l][m] + v[j][counter]
                counter += 1

            


    #regression
    else:
      for i in range(loops):
        for j in range(len(parts)):
          #used to hold the correct and the guessed classes
          values = []
          actual = []

          #the first particle
          if i == 0 and j == 0:
            for k in range(len(training_df)):
              parts[j].forwardProp(training_df.iloc[k,0:-1].values.astype('float'), dataType)
              actual.append(training_df.values[k,-1])
              values.append(uniques[parts[j].values[-1].index(max(parts[j].values[-1]))])

            lossValues = lss.Loss()
            lossValues.calculateReg(values, actual)
            
            #used to hold all values of the weights themselves
            temp = []
            for k in range(len(parts[j].mlp_init)):
              for l in range(len(parts[j].mlp_init[k])):
                for m in range(len(parts[j].mlp_init[k][l])):
                  temp.append(parts[j].mlp_init[k][l][m])

            #sets the global best and the personal best because it is the first loop
            gb = [lossValues.mse, temp]
            pb.append([lossValues.mse, temp])

          #the first loop through the particles
          elif i == 0:
            for k in range(len(training_df)):
              parts[j].forwardProp(training_df.iloc[k,0:-1].values.astype('float'), dataType)
              actual.append(training_df.values[k,-1])
              values.append(uniques[parts[j].values[-1].index(max(parts[j].values[-1]))])

            lossValues = lss.Loss()
            lossValues.calculateReg(values, actual)

            #used to hold all values of the weights themselves
            temp = []
            for k in range(len(parts[j].mlp_init)):
              for l in range(len(parts[j].mlp_init[k])):
                for m in range(len(parts[j].mlp_init[k][l])):
                  temp.append(parts[j].mlp_init[k][l][m])

            if gb[0] > lossValues.mse:
              gb = [lossValues.mse, temp]
            
            pb.append([lossValues.mse, temp])

          #second or more loops through the particles
          else:
            #runs the model with the given training set
            for k in range(len(training_df)):
              parts[j].forwardProp(training_df.iloc[k,0:-1].values.astype('float'), dataType)
              actual.append(training_df.values[k,-1])
              values.append(uniques[parts[j].values[-1].index(max(parts[j].values[-1]))])

            #calculates loss on that model
            lossValues = lss.Loss()
            lossValues.calculateReg(values, actual)

            #check for personal best mse score
            if pb[j][0] > lossValues.mse:
              #used to hold all values of the weights themselves
              temp = []
              for k in range(len(parts[j].mlp_init)):
                for l in range(len(parts[j].mlp_init[k])):
                  for m in range(len(parts[j].mlp_init[k][l])):
                    temp.append(parts[j].mlp_init[k][l][m])

              pb[j] = [lossValues.mse, temp]

            #check for global best mse score
            if gb[0] > lossValues.mse:
              gb = [lossValues.mse,temp]
              print(i, gb[0])
          
            


        for j in range(len(parts)):
          counter = 0
          #update velocity and position
          for k in range(len(parts[j].mlp_init)):
            for l in range(len(parts[j].mlp_init[k])):
              for m in range(len(parts[j].mlp_init[k][l])):
                r_1 = random.uniform(0,1)
                r_2 = random.uniform(0,1)
                v[j][counter] = (.000001*v[j][counter]) + c_1*r_1*(pb[j][1][counter] - parts[j].mlp_init[k][l][m]) + c_2*r_2*(gb[1][counter] - parts[j].mlp_init[k][l][m])
                parts[j].mlp_init[k][l][m] = parts[j].mlp_init[k][l][m] + v[j][counter]
                counter += 1

#---------Testing-Area------------
# import requests

# with open('loss.py', 'w') as f:
#     r = requests.get('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/code/loss.py?raw=true')
#     f.write(r.text)

# import loss as lss

# models = []
# hiddenArray = [3,2]

# for i in range(30):
#   model = Model()
#   model.run(2, hiddenArray, 4)
#   models.append(model)

# training = pd.DataFrame(columns=['length','height','Class'], data=[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[1,2,1],[2,3,2],[3,2,3],[4,2,4],[1,3,1],[2,1,2],[3,4,3],[4,1,4],[1,4,1],[2,4,2],[3,1,3],[4,1,4]])
# testing = pd.DataFrame(columns=['length','height','Class'],data=[[1,1,1],[3,3,3]])

# topModel = Model()
# topModel.algPSO(models, 1, training, testing.iloc[:,:-1], testing)
