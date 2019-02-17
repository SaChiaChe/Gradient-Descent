# Decision Stump
A practice of gradient descent.

## How to run

Start the program
```
python GradientDescent.py <TrainDataFile> <TestDataFile> <LearningRate> <Iterations>
```
```
python StochasticGradientdescent.py <TrainDataFile> <TestDataFile> <LearningRate> <Iterations>
```
```
python GDvsSGD.py <TrainDataFile> <TestDataFile> <GDLearningRate> <GDIterations> <SGDLearningRate> <SGDIterations>
```

### GradientDescent.py

Run gradient descent on the train dataset with the desired learing rate and iterations, and output
the error for train dataset and error for test dataset.

### StochasticGradientdescent.py

Same as GradientDescent.py, but this time only picks one sample to update rather than updating with the entire train dataset, this is called "stochastic" gradient descent. It should be much faster than GradientDescent.py, speedup by a factor of the size of the train dataset.

### GDvsSGD.py

A comparison between gradient descent (GD) and stochastic gradient descent (SGD).
We could see that the stochastic gradient decent is unstable and jumps up and down all the time, and gradient descent is smoothe, both have the same trend of getting lower and lower errors.

## Built With

* Python 3.6.0 :: Anaconda custom (64-bit)

## Authors

* **SaKaTetsu** - *Initial work* - [SaKaTetsu](https://github.com/SaKaTetsu)