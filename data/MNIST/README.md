# MNIST in CSV

* The original dataset comes from Yann LeCun in this [website](http://yann.lecun.com/exdb/mnist/).
* The CSV version comes from the Kaggle competition [here](https://www.kaggle.com/oddrationale/mnist-in-csv/downloads/mnist-in-csv.zip/2).
    * The zip file here are compressed on MBP with command `zip FILENAME.zip FILENAME.csv`

## Data generation Process

* Requirement:
    * The length should not exceed 32
    * binary case
        * half contains 0 (label `0`) and the other half not (label `1`)
        * In sequences with label `1`, there is one zero (multiple zeros are also possible, but the number would be hard to interpret if there is no correspnding tasks)
    * muticlass case
        * the number of different labels (`{0, 1, 2}`) is balanced

* Algorithm:
    * Select the label of for all the sequences sequence
        * `np.random.choice([0, 1, 2], size=10000, replace=True)`
    * Generate the lengths of sequences, ranging from 0 to 32 (mean 20, std 5, clipped for 3 and 32)
    * Select the rest digits in the sequence uniformly at random from the pool of digits uniformly at random

 * Shape of the new dataset
    * Follow the tensorflow [RNN API](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN#input_shape): `(batch, timesteps, features)`
    * Training set: `(10000, 32, 784)`
    * Test set: `(1000, 32, 784)`

 * Utility

    * Visualize the sequence data and the labels


## Conterfactual Data generation

* Follow the supervised-to-bandit conversion, have to figure out a predefined logging policy. 
