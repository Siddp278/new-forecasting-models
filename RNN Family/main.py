import torch.nn as nn

from utils import *
from model import ConvRNN
from train import *


data_source = [
    "sml2010/NEW-DATA-1.T15.txt",
    "sml2010/NEW-DATA-2.T15.txt"
]

target = '3:Temperature_Comedor_Sensor'
features = [
    '3:Temperature_Comedor_Sensor',
 '4:Temperature_Habitacion_Sensor',
 '5:Weather_Temperature',
 '6:CO2_Comedor_Sensor',
 '7:CO2_Habitacion_Sensor',
 '8:Humedad_Comedor_Sensor',
 '9:Humedad_Habitacion_Sensor',
 '10:Lighting_Comedor_Sensor',
 '11:Lighting_Habitacion_Sensor',
 '12:Precipitacion',
 '13:Meteo_Exterior_Crepusculo',
 '14:Meteo_Exterior_Viento',
 '15:Meteo_Exterior_Sol_Oest',
 '16:Meteo_Exterior_Sol_Est',
 '20:Exterior_Entalpic_2',
 '21:Exterior_Entalpic_turbo',
 '22:Temperature_Exterior_Sensor']

train_size = 3200
val_size = 400
depth = 90
batch_size = 128

X, y, shapes = data_sequence(data_source, train_size, val_size, depth, features, target)
# print(shapes)

X, y, _ = preprocess(X, y, bias=1e-9)

train_loader, val_loader, test_loader = dataset_generator(X, y, batch_size)

# # testing the model:
# for batch_x, batch_y in train_loader:
#     x = batch_x
#     break

# model_type = "RNN"
# model = ConvRNN(model_type, X[0].shape[2], X[0].shape[1], 1, n_channels1=128, n_channels2=128, n_channels3=128,
#                 n_units1=128, n_units2=128, n_units3=128).cuda()

# out = test_model(x)
# print(out)


model_type = "RNN"
# model_type = "GRU"
# model_type = "LSTM"
model = ConvRNN(model_type, X[0].shape[2], X[0].shape[1], 1, n_channels1=128, n_channels2=128, n_channels3=128,
                n_units1=128, n_units2=128, n_units3=128).cuda()

epochs = 1000
loss_fn = nn.MSELoss()
patience = 150
learning_rate = 0.001

# No Regularization used on all models
# train(model, model_type, epochs, loss_fn, patience, learning_rate, train_loader, val_loader, shapes)

# evaluate(test_loader, model, loss_fn, _[2], _[3])

# L1 Regularization used on all models
# train(model, model_type, epochs, loss_fn, patience, learning_rate, train_loader, val_loader, shapes, l1=True)

# evaluate(test_loader, model, loss_fn, _[2], _[3])

# L2 Regularization used on all models
# train(model, model_type, epochs, loss_fn, patience, learning_rate, train_loader, val_loader, shapes, l2=True)

# evaluate(test_loader, model, loss_fn, _[2], _[3])

# Elnet Regularization used on all models
train(model, model_type, epochs, loss_fn, patience, learning_rate, train_loader, val_loader, shapes, ElNet=True)

evaluate(test_loader, model, loss_fn, _[2], _[3])
