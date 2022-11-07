#------------------------Header-------------------------

#Code by: Isaac Boyd, James Lucas 

##Code For: Preprocessing
##Completed: 11-7-2022
##References: NA

#-----------------------imports-------------------------

import pandas as pd
import math
import random
import numpy as np

#------------------------Class--------------------------
class Preprocessing:

  #initialization
  def __init__ (self):
    self.df = pd.DataFrame
    self.folds = []
    self.tuning = []
    self.value = int()
    self.oneHotDict = {}
    self.classes = []

  #method that applies a z-score noramlization to the data
  def normalize(self):
    smalldf = self.df.iloc[:,:len(self.df.iloc[0])-1]

    #makes an array used to hold all of the means for each attribute
    meanArray = np.zeros(len(smalldf.iloc[0]))
    #loop that calculates the mean 
    for i in range(len(smalldf)):
      for j in range(len(smalldf.iloc[i])):
        meanArray[j] = meanArray[j] + smalldf.iloc[i][j]

    for i in range(len(meanArray)):
      meanArray[i] = meanArray[i]/len(smalldf)

    #makes an array used to hold all of the standard deviations for each attribute
    deviation = np.zeros(len(smalldf.iloc[0]))
    for i in range(len(smalldf)):
      for j in range(len(smalldf.iloc[0])):
        deviation[j] = deviation[j] + (smalldf.iloc[i][j] - meanArray[j])**2


    for i in range(len(deviation)):
      tempDev = deviation[i]/(len(smalldf)-1)
      deviation[i] = math.sqrt(tempDev)

    #changes the original dataframe values to the new z-score values
    #also leaves the class/regression-values as the original value 
    for i in range(len(smalldf)):
      for j in range(len(smalldf.iloc[i])):
        self.df.iat[i,j] = (smalldf.iloc[i][j] - meanArray[j])/deviation[j]
        if pd.isnull(self.df.iloc[i][j]):
          self.df.iat[i,j] = 0


  #method that allows the user to choose the dataset, and preprocesses that said data set.
  def process(self):
    #gets user input 
    DataNumber = input("Enter a Number(1-6) for data selection: ")

    #import dataset 1
    if DataNumber == '1':
      df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/Data/breast-cancer-wisconsin.csv?raw=true')
      print("Using Breast Cancer data (Classification)")

      #removes the rows with question marks
      df = df[~(df=='?').any(axis=1)].reset_index(drop=True)

      #makes all of the values integers
      for i in range(len(df)):
        for j in range(len(df.iloc[0])):
          df.iloc[i][j] = int(df.iloc[i][j])

      self.value = 0
      self.df = df

    #import dataset 2
    elif DataNumber == '2':
      df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/Data/glass.csv?raw=true')
      print("Using Glass data (Classification)")

      self.value = 0
      self.df = df

    #import dataset 3
    elif DataNumber == '3':
      df = pd.read_csv('https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/Data/soybean_small.csv?raw=true')
      print("Using Soybean data (Classification)")

      self.value = 0
      self.df = df

    #import dataset 4
    elif DataNumber == '4':
      df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/Data/abalone.csv?raw=true")
      print("Using Abalone data (Regression)")

      oneEncodeList = []

      #loops through the whole data set to change M F and I to one hot encoding
      for i in range(len(df)):
        if df.iloc[i,0] == 'M':
          oneEncodeList.append([1,0,0])
        elif df.iloc[i,0] == 'F':
          oneEncodeList.append([0,1,0])
        else:
          oneEncodeList.append([0,0,1])

      adding = pd.DataFrame(oneEncodeList, columns =['Male', 'Female', 'Other'])
      #adds the one hot encoding onto the end of the data set
      df = df.join(adding)
      #takes the sex column out of the data set
      df.drop(columns=["Sex"], inplace = True)
      df = df.sort_values(by=['Rings'])

      #moves the rings column to the end of the data set
      classColumn = df.pop("Rings")

      df.insert(len(df.columns), "Rings", classColumn)
      df = df.reset_index()

      self.value = 1
      self.df = df

    #import dataset 5
    elif DataNumber == '5':
      df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/Data/machine.csv?raw=true")
      print("Using Machine data (Regression)")

      #takes out the 'VendorName' and 'ModelName' columns
      df.drop(columns=["VendorName"],  inplace = True)
      df.drop(columns=["ModelName"],  inplace = True)
      df = df.sort_values(by=['PRP'])

      self.value = 1
      self.df = df

    #import dataset 6
    elif DataNumber == '6':
      df = pd.read_csv("https://github.com/IsaacBoyd2/ActualFactualML/blob/main/Project3/Data/forestfires.csv?raw=true")
      print("Using Forest Fires data (Regression)")

      #sets key values for the months and days
      months = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
      days = {'sun':1,'mon':2,'tue':3,'wed':4,'thu':5,'fri':6,'sat':7}

      #changes the values in the data set from strings to the associated value
      for i in range(len(df)):
        df = df.replace({"month":months})
        df = df.replace({"day":days})
        df = df.sort_values(by=['area'])

      self.value = 1
      self.df = df

    #used to say that the number entered was incorrect
    else:
      print("That is not a valid value for picking the data set.")
      process(self)

  #function used to make the tuning fold, and 'foldNumber' amount of folds
  def fold(self):
    #number of folds (mostly 10 in this case)
    foldNumber = 10
    fold = []
    foldTotal = []
    randomList = random.sample(range(len(self.df)), len(self.df))

    #folds depending on categorical or regression data

    #categorical
    if self.value == 0:

      #gets the size of each class
      classArray = self.df['Class']
      classes = self.df['Class'].unique()
      classSize = np.zeros((len(classes),1))
      for i in range(len(self.df)):
        for j in range(len(classes)):
          if classArray.iloc[i] == classes[j]:
            classSize[j] = classSize[j] + 1

      total = len(self.df)
      classPercent = np.zeros((len(classes),1))

      #gets the percent of the class from the whole data set
      for i in range(len(classSize)):
        classPercent[i] = classSize[i]/total

      total = total*.1
      classAmount = np.zeros((len(classes),1))

      #gets the amount that we need from each class for the tuning array
      for i in range(len(classSize)):
        classAmount[i] = round(float(classPercent[i] * total))

      dfHolder = self.df.copy()

      #used to get the 'foldNumber' amount of folds
      for m in range(foldNumber):
        fold = []
        #loops through the amount of classes
        for i in range(len(classAmount)):
          #loops for as many of each class we need
          for j in range(int(classAmount[i][0])):
            #loops through the randomList in order to find classes that are needed
            for k in range(len(randomList)):
              if classArray.iloc[randomList[k]] == classes[i]:
                fold.append(dfHolder.iloc[randomList[k]])
                dfHolder = dfHolder.drop(randomList[k])
                dfHolder = dfHolder.reset_index(drop=True)
                classArray = classArray.drop(randomList[k])
                classArray = classArray.reset_index(drop=True)
                #changes all of the indexes greater than the one that we are taking out
                for l in range(len(randomList)):
                  if randomList[l] > k:
                    randomList[l] = randomList[l] - 1
                randomList.remove(k)
                break
        foldTotal.append(fold)

      self.folds = foldTotal


    #regression
    else:
      #sets values for looping through the data
      index = 0
      total = len(self.df)
      percentage = total*.1
      percentage = round(len(self.df)/percentage)
      foldAmount = round(total*.1)
      dfFolds = []
      lastIndex = 0

      #loops through the amount of folds, then make a 'cut' at each 'class'
      for i in range(foldNumber):
        if lastIndex < len(self.df):
          dfFolds.append(self.df.iloc[(lastIndex):lastIndex+foldAmount])
          lastIndex = lastIndex + foldAmount
        else:
          dfFolds.append(self.df.iloc[lastIndex:len(self.df)])

      #shuffles the classes
      for i in range(len(dfFolds)):
        dfFolds[i] = dfFolds[i].sample(frac=1).reset_index()


      foldTotal = []
      #loops for the amount of folds
      for k in range(foldNumber):
        fold = []
        #goes through each class in order to stratisfy the data
        for i in range(len(dfFolds)):
          #gives each fold the correct amount of points
          for j in range(round(foldAmount/foldNumber)):
            if j < len(dfFolds[i]):
              fold.append(dfFolds[i].iloc[j])
              dfFolds[i] = dfFolds[i].drop(j)
          dfFolds[i] = dfFolds[i].reset_index(drop=True)
        
        foldTotal.append(fold)
        
      self.folds = foldTotal

  #makes the classes, for classification, into one hot encoded values
  def oneHot(self):
    #gets the unique classes
    classes = self.df['Class'].unique()
    #makes a dictionary that holds the classes and their associated one hot encoded value
    oneHotDict = {}
    for i in range(len(classes)):
      oneHotDict.update({classes[i]: []})
      for j in range(len(classes)):
        if j != i:
          oneHotDict[classes[i]].append(0)
        else:
          oneHotDict[classes[i]].append(1)

    #saves these values to the Preprocessing object
    self.classes = classes
    self.oneHotDict = oneHotDict
