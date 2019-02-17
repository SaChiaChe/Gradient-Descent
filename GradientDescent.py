import sys
import numpy as np
from func.ReadData import *
from func.Calculations import *

if len(sys.argv) != 5:
	print("Format: python GradientDescent.py <TrainDataFile> <TestDataFile> <LearningRate> <Iterations>")
	exit(0)

TrainDataFilePath, TestDataFilePath = sys.argv[1], sys.argv[2]
Train_X, Train_Y = ReadData(TrainDataFilePath)
Test_X, Test_Y = ReadData(TestDataFilePath)
Dimension = len(Train_X[0])

LearningRate = float(sys.argv[3])
IterationCount = int(sys.argv[4])
Weight = np.array([0] * Dimension)

for Iteration in range(IterationCount):
	Gradient = CalGradient(Weight, Train_X, Train_Y)
	Weight = Weight - LearningRate * Gradient

E_in = CalError(Weight, Train_X, Train_Y)
E_out = CalError(Weight, Test_X, Test_Y)

print("E_in:", E_in)
print("E_out:", E_out)
print("W:", Weight)