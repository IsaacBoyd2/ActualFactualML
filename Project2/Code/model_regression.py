# -*- coding: utf-8 -*-
"""model_regression

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12QGRaGIHb9rWQOmw1KH_s7ZCZ200Xp61
"""

#-----------------------imports-------------------------
import pandas as pd
import numpy as np
import requests
from scipy import stats as st
from sympy import symbols, Eq, Sum, solve
import math

#Hyper parameters
k_nn = 5

df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/abalone.csv?raw=true")

df_matrix = pd.DataFrame(np.nan, index=range(len(df)), columns = range(len(df)))

#df = df.drop(0)
df = df.reset_index()
df2 = df.iloc[: , 1:-1]

df2 = df2[0:10]


for counts, i in enumerate(df2['Sex']):
  if i =='M':
    df2['Sex'].iloc[counts] = 1
  elif i == 'F':
    df2['Sex'].iloc[counts] = 2
  else:
    df2['Sex'].iloc[counts] = 0   #Need fo find out the other sexes


for count1 in range(len(df2)):
    base = df2.iloc[count1]
    for count2 in range(len(df2)):
      dist1 = []
      for count3 in range(len(df2.columns)):
        dist1.append((base[count3] - df2.iloc[count2][count3])**2)
      
      summation = sum(dist1)
      distance = np.sqrt(summation)

      df_matrix.loc[count1, count2] = distance 

    comparison_array = np.array(df_matrix.loc[count1])
    k = k_nn+1
    index = np.argpartition(comparison_array, k)
    reduced_idx = index[:k]

    reduced_list = reduced_idx.tolist()

    reduced_list.remove(count1)

    #print(reduced_list)

    x = symbols('x')


    desicion_list = []
    for i in reduced_list:
      desicion_list.append(df['Rings'][i])

    print(desicion_list)


    #eq1 = Eq(4*x + 2)


    #reduced list has the values of our k_nearest neighbors inside of them
    #w_i is a gaussian that assigns a wieght to the function depending on how far away a given value is from a neighbor
    #w_i is our denominator and i "y_i" * w_i is our numerator. 

    denominator = []
    numerator = []

    for i in desicion_list:
      weight_holder = []
      for ii in desicion_list:
        if i != ii:
          w_i_part = 2.72**(-(i-ii)**2)
          weight_holder.append(w_i_part)
      w_i = sum(weight_holder)
      denominator.append(w_i)
      y_i_w_i = w_i * i
      numerator.append(y_i_w_i) 
    
    f = sum(numerator)/sum(denominator)
    print(f)

w_i

print(sum(equation_list))

df

df2

df_matrix

df2

df[0:50]

#-----------------------imports-------------------------
import pandas as pd
import numpy as np
import requests
from scipy import stats as st

#Hyper parameters
k_nn = 3

df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project2/Data/abalone.csv?raw=true")

df_matrix = pd.DataFrame(np.nan, index=range(len(df)), columns = range(len(df)))


df = df.drop(0)
df = df.reset_index()
df2 = df.iloc[: , 2:-1]

decision = []

for count1 in range(len(df2)):
    base = df2.iloc[count1]
    for count2 in range(len(df2)):
      dist1 = []
      for count3 in range(len(df2.columns)):
        dist1.append((base[count3] - df2.iloc[count2][count3])**2)
      
      summation = sum(dist1)
      distance = np.sqrt(summation)

      df_matrix.loc[count1, count2] = distance 

    comparison_array = np.array(df_matrix.loc[count1])
    k = k_nn+1
    index = np.argpartition(comparison_array, k)
    reduced_idx = index[:k]

    reduced_list = reduced_idx.tolist()

    reduced_list.remove(count1)

    majority =[]
    for i in reduced_list:
      #print(df['Class'][i])
      majority.append(df['Class'][i])

    class_decision = st.mode(majority)
    

    #print(class_decision[0])
    decision.append(class_decision)

counts = 0
for i in range(len(decision)):
  #print(decision[i][0])
  #print(df['Class'][i])
  if decision[i][0] == df['Class'][i]:
    counts += 1

print(counts)
print(counts/len(df))


#Next we get the smallest values in the rows     Done
#Then we find the associated slice for the mins.  Done
#Then we find the associated classes  Done
#Then we do a majority vote and classify. Done
#Then we find loss. Done



#For regression take gaussians for every nearest neighbor.
#sum all the area underneath the graph
#pick the highest point on the graph to be your regressed value.


#For each slice. Find the distance between all other points. 

#Slice 1: dif 1,dif2, ... , dif99
#Slide 2:
#Slide 3:
#...
#Slice 99: