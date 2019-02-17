import math
import numpy as np

def CalGradient(W, X, Y):
	Temp = 0
	N = len(X)
	for i in range(N):
		Xn, Yn = X[i], Y[i]
		Temp += (-Yn * Xn) / (1 + math.exp(Yn * np.dot(W, Xn)))
	return Temp / N

def CalStochasticGradient(W, X, Y, I):
	N = len(X)
	Xn, Yn = X[I % N], Y[I % N]
	return (-Yn * Xn) / (1 + math.exp(Yn * np.dot(W, Xn)))

def CalError(W, X, Y):
	N = len(X)
	ErrorCount = 0
	for i in range(N):
		ErrorCount += 0 if np.dot(W, X[i])*Y[i] >= 0 else 1
	return ErrorCount / N