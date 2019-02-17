def ReadData(File):
	import numpy as np

	X, Y = [], []

	f = open(File, "r")
	fl = f.readlines()
	for line in fl:
		Data = line.split()
		X.append([1.0] + [float(x) for x in Data[0:-1]]) 
		Y.append(int(Data[-1]))
	f.close()

	return np.array(X), np.array(Y)