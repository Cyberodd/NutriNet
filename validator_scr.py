import numpy as np
import pickle as pickle
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

train_file = 'C:/Users/Kanverse/Documents/train1_scaled.csv'
validation_file = 'C:/Users/Kanverse/Documents/validation1_scaled.csv'
output_model_file = 'model_val.pkl'

hidden_size = 100
epochs = 500
continue_epochs = 10
validation_proportion = 0.15

# load data, join train and validationSupp

train = np.loadtxt( train_file, delimiter = ',' )
validation = np.loadtxt( validation_file, delimiter = ',' )
train = np.vstack(( train, validation ))

x_train = train[:,1:15]
y_train = train[:,-1]
y_train = y_train.reshape( -1, 1 )

input_size = x_train.shape[1]
target_size = y_train.shape[1]

# prepare dataset

ds = SDS( input_size, target_size )
ds.setField( 'input', x_train )
ds.setField( 'target', y_train )

# init and train

net = buildNetwork( input_size, hidden_size, target_size, bias= True )
trainer = BackpropTrainer( net,ds )

train_mse, validation_mse = trainer.trainUntilConvergence( verbose = True, validationProportion = validation_proportion,
                                                           maxEpochs = epochs, continueEpochs = continue_epochs )

for i in range( epochs ):
    mse = trainer.train()
    rmse = sqrt( mse )
    print ("training RMSE, epoch {}: {}".format( i + 1, rmse ))

pickle.dump( net, open( output_model_file, 'wb'))
