#------------------------Header-------------------------

#Code by: Isaac Boyd, James Lucas 

##Code For: loss (F1, precision, recall, mse, and mae)
##Completed: 9-29-2022
##References: NA

#-----------------------imports-------------------------

import pandas as pd
import math
import random
import numpy as np

#------------------------Class--------------------------
class Loss:

  #initialization
  def __init__(self):
    self.prec = []
    self.rec = []
    self.error = []
    self.F1 = int()
    self.mse = int()
    self.mae = int()
  
  #used to calculate precision, recall, and F1 for classification data sets
  def calculate(self, classes, pred, facts):
    #creates a confusion Matrix of correct size
    confusionMat = np.zeros([len(classes),len(classes)])

    #populates the confusion matrix that will be used for precision, recall, and F1 calculations.
    for i in range(len(facts)):
      indHorz = 0
      indVert = 0
      print('Classes', classes, "predictions", pred, "actual", facts)
      #checks for position in the confusion matrix
      for k in range(len(classes)):
        if classes[k] == pred[i]:
          indVert = k
        if classes[k] == facts[i]:
          indHorz = k
      #adds an occurance in the confusion matrix
      confusionMat[indHorz, indVert] += 1

      print("ConfusionMatrix: \n", confusionMat)

    #calculates precision and recall
    for i in range(len(confusionMat)):
      truePos = 0
      falsePos = 0
      falseNeg = 0
      #loops through the confusion matrix for popluation
      for j in range(len(confusionMat[i])):
        #true positive
        if i == j:
          truePos = confusionMat[i][j]
        #[i][j] is a false positive, and [j][i] is a false negative
        else:
          falsePos = falsePos + confusionMat[i][j]
          falseNeg = falseNeg + confusionMat[j][i]
        
      #precision and recall calculations
      if (truePos+falsePos) != 0:
        precision = truePos/(truePos+falsePos)
      else:
        precision = 0

      if (truePos+falseNeg) != 0:
        recall = truePos/(truePos+falseNeg)
      else:
        recall = 0


      #assigns the precision and recall to the respective self arrays
      self.prec.append(precision)
      self.rec.append(recall)

    #gets the average precision and recall, then calculates the F1 score
    avgPrec = 0
    avgRec = 0
    for i in range(len(self.prec)):
      avgPrec = avgPrec + self.prec[i]
      avgRec = avgRec + self.rec[i]
    avgPrec = avgPrec/len(self.prec)
    avgRec = avgRec/len(self.rec)

    #assigns the F1 score to the object
    self.F1 = 2*((avgPrec*avgRec)/(avgPrec+avgRec))

  #used to calculate mse and mae for regression data sets
  def calculateReg(self, pred, facts):
    distance = float(0)
    distanceabs = float(0)
    total = 0
    #goes through all of our predictions and compares them to the actual result
    for i in range(len(pred)):
      distance = distance + ((float(facts[i]) - float(pred[i]))**float(2))
      distanceabs = distanceabs + abs((float(facts[i]) - float(pred[i])))
      #saves all of the errors for each point
      self.error.append((float(facts[i]) - float(pred[i]))**float(2))
      total = total+1

    #makes sure that our predictions and actual results are not empty
    if len(facts) > 0:
      self.mse = distance/total
      self.mae = distanceabs/total
    else:
      self.mse = 0
