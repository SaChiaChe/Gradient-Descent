import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from func.ReadData import *
from func.Calculations import *

if len(sys.argv) != 7:
	print("Format: python GDvsSGD.py <TrainDataFile> <TestDataFile> <GDLearningRate> <GDIterations> <SGDLearningRate> <SGDIterations>")
	exit(0)

TrainDataFilePath, TestDataFilePath = sys.argv[1], sys.argv[2]
Train_X, Train_Y = ReadData(TrainDataFilePath)
Test_X, Test_Y = ReadData(TestDataFilePath)
Dimension = len(Train_X[0])

GDLearningRate, GDIterationCount = float(sys.argv[3]), int(sys.argv[4])
SGDLearningRate, SGDIterationCount = float(sys.argv[5]), int(sys.argv[6])
InitWeight = np.array([0.] * Dimension)

#GD
GDWeight = InitWeight
GDTrackEin, GDTrackEout = [], []
for Iteration in range(GDIterationCount):
	Gradient = CalGradient(GDWeight, Train_X, Train_Y)
	GDWeight = GDWeight - GDLearningRate * Gradient
	E_in = CalError(GDWeight, Train_X, Train_Y)
	E_out = CalError(GDWeight, Test_X, Test_Y)
	GDTrackEin.append(E_in)
	GDTrackEout.append(E_out)

#SGD
SGDWeight = InitWeight
SGDTrackEin, SGDTrackEout = [], []
for Iteration in range(SGDIterationCount):
	Gradient = CalStochasticGradient(SGDWeight, Train_X, Train_Y, Iteration)
	SGDWeight = SGDWeight - SGDLearningRate * Gradient
	E_in = CalError(SGDWeight, Train_X, Train_Y)
	E_out = CalError(SGDWeight, Test_X, Test_Y)
	SGDTrackEin.append(E_in)
	SGDTrackEout.append(E_out)	


### PLOTTING ###
#plot E_in
# GD
plt.figure("E_in")
GDaxisY = list(range(GDIterationCount))
plt.plot(GDaxisY, GDTrackEin, label = "GD")
# SGD
SGDaxisY = list(range(SGDIterationCount))
plt.plot(SGDaxisY, SGDTrackEin, label = "SGD")
# Labels
plt.ylabel('E_in')
plt.xlabel('Iteration')
plt.legend()

#plot E_out
# GD
plt.figure("E_out")
GDaxisY = list(range(GDIterationCount))
plt.plot(GDaxisY, GDTrackEout, label = "GD")
# SGD
SGDaxisY = list(range(SGDIterationCount))
plt.plot(SGDaxisY, SGDTrackEout, label = "SGD")
# Labels
plt.ylabel('E_out')
plt.xlabel('Iteration')
plt.legend()

#Show plots
plt.show()