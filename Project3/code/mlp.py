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
    
  def forwardProp(self,input, classNumber):      #potentially need to do something for just the input layers
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
          #print("SUMMATION: ", summation)
          sigmoid = 1/(1+round(math.e**(-summation),8))    #sigmoid function
          layer_outputs.append(sigmoid) #append for each input

        values.append(layer_outputs) #append all the outputs. (this will be what is "inside" of each node)
        #print(values)

      #output layer

      elif classNumber == 1:
        #print(self.mlp_init)
        #print(values)
        for i in range(1):
          #print(len(self.mlp_init[-1]))
          l = []
          for k in range(len(values[-1])):   #for every xi
            l.append(float(values[-1][k])*float(self.mlp_init[-2][k][i]))  #do xiwi
          summation = sum(l)
        
        output = summation

        self.output = output

        #print('output',output)


        #Decision Circuit
      else:
        layer_outputs = []
        for i in range(len(self.mlp_init[-1][0])):
          #print(len(self.mlp_init[-1]))
          l = []
          for k in range(len(values[-1])):   #for every xi
            l.append(float(values[-1][k])*float(self.mlp_init[-1][k][i]))  #do xiwi
          summation = sum(l) #Sum of all xiwis
          sigmoid = 1/(1+math.e**(-summation))    #sigmoid function
          layer_outputs.append(sigmoid) #append for each input

        values.append(layer_outputs) #append all the outputs. (this will be what is "inside" of each node)
      
      if classNumber == 0:

        sum_of_soft = []
        for i in values[-1]:
          softmax1 = (math.e**i)
          sum_of_soft.append(softmax1)
          
        the_sum_of_soft = sum(sum_of_soft)

        output_values = []

        for i in values[-1]:
          softmax2 = (math.e**i)/the_sum_of_soft
          output_values.append(softmax2)

        values[-1] = output_values

      
    self.values = values

  def Back_Prop(self,eta,classNumber,actual,output_size):  
    #print(self.mlp_init)
    #print(self.mlp_init)
    deltas=[]  
    for x in range(len(self.mlp_init)):
      deltas.append([])

    #print(deltas)
    #deltas[2][] = 1
    #deltas[1].append(1)
    #deltas[2].append(2)
    #print(deltas)

    counter = 0
    for i in reversed(range(len(self.mlp_init))):   #go through every layer backwards
      #print(self.mlp_init[i])
      farthest_layer_right = self.mlp_init[i]    #grab the last layer
      #print(len(farthest_layer_right)) 

      if i == len(self.mlp_init)- 1:    #output layer
        #print(len(self.mlp_init))
        #print('hello :)')
        if classNumber == 1:
          diff = self.output - actual                   #delta is actual - predicted * derivative of the actication function. So for the sigmoid layers this would be (ri-yi)(oj(1-oj)) and linear it would just be (ri-yi) * possibly C

          deltas[counter].append(diff)
          
          #print(self.mlp_init[-1])

          #self.mlp_init[i][len(self.mlp_init[-1][-1])][self.mlp_init[-1][-1][-1]] = self.mlp_init[i][len(self.mlp_init[-1][-1])][self.mlp_init[-1][-1][-1]] + eta*diff*self.output
          #print(diff)
        else:
          actualClass = actual[1]
          actualOneHot = actual[0]

          for j in range(len(self.values[i])):
            diff = actualOneHot.get(actualClass)[j] - self.values[i][j]
            deltas[0].append(diff)

      else:

        print(len(farthest_layer_right))
        for j in range(len(farthest_layer_right[0])): 
          #print(j) 
          #print(i)
          #print(self.values)
          #print(self.values[0][3])

          xi = self.values[i+1][j]


          #print('asdfasdf',self.values)

          #weight_sum = 0
          sumwih_deltai = 0

          for l in range(len(self.mlp_init[i+1][0])):
            #print(len(self.mlp_init[i+1][0]))

            #print(self.mlp_init[i+1])
            if l < len(deltas[counter]):
              print("\n\nLen of Deltas(coutner/l): ", len(deltas), len(deltas[counter]), "counter/l: ", counter, l)
              deltai = deltas[counter][l]
              for m in range(len(farthest_layer_right)): 
                weight_s = self.mlp_init[i][m][l]

              sumwih_deltai += weight_s*deltai
          try:
            delcalc = sumwih_deltai*(xi)*(xi-1) 
            deltas[counter+1].append(delcalc)
          except:
            pass


        

        counter = counter + 1  
    #print('Fowards')
    #print(self.mlp_init)
    #print(self.values)
    #print(deltas)
    deltas.reverse()
    #print(self.mlp_init)

    for i in range(len(self.mlp_init)):
      layer = self.mlp_init[i]
      for j in range(len(layer)):
        neuron = layer[j]
        for k in range(len(neuron)):
          print(i)
          print(j)
          print(k)
          print(len(self.mlp_init[i][j]), len(deltas), len(self.values[i]))
          if k < len(self.mlp_init[i][j]) and j < len(self.values[i]) and k < len(deltas[i]):
            self.mlp_init[i][j][k] = self.mlp_init[i][j][k] + eta*deltas[i][k]*self.values[i][j]


      #print(counter)
            #delta = learning_rate* diff *
          #else: #hidden layers
          #  self.mlp_init[i][j][k] = k + eta*delta*self.values[i][j]
    # print(self.mlp_init)
#we need weight, xji, oj (for error), (oj(1-oj)), actual-predicted
#weight = weight - learning_rate * error * input


# learning_rate = 0.5
# modeling = Model()
# modeling.run(3, [4,3,2], 1)
# input = [1,0.5,0.75]
# modeling.forwardProp(input,1)
# modeling.Back_Prop(learning_rate)
