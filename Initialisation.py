#Import Dependencies
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

#Load Dataset
train_X, train_Y, test_X, test_Y = load_dataset()

def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
  grads = {}
  costs = []
  m = X.shape[1]
  layers_dims = [X.shape[0], 10, 5, 1]
  
  if initialization == "zeros":
    parameters = initialize_parameters_zeros(layers_dims)
  elif initialization == "random":
    parameters = initialize_parameters_random(layers_dims)
  elif initialization =="he":
    parameters = initialize_parameters_he(layers_dims)
  
  for i in range(0, num_iterations):
    a3, cache = forward_propagation(X, parameters)
    
    cost = compute_loss(a3, Y)
    
    grads = backward_propagation(X, Y, cache)
    
    parameters = update_parameters(parameters, grads, learning_rate)
    
    if print_cost and i % 1000 == 0:
      print("cost after iteration {}: {}" .format(i, cost))
      costs.append(cost)
      
  plt.plot(costs)
  plt.xlabel("iterations (per hundreds)")
  plt.ylabel("cost")
  plt.title("Learning rate = " + str(learning_rate))
  plt.show()
  
  return parameters
  
#Zero Initialization
def initialize_parameters_zero(layers_dims):
  parameters = {}
  L = len(layers_dims)
  
  for l in range(1, L):
    parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
    parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    
  return parameters

parameters = initialize_parameters_zeros([3,2,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

parameters = model(train_X, train_Y, initialization = "zeros")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

print("predictions_train = " + str(predictions_train))
print("predictions_test = " + str(predictions_test))
#All the outputs are zeros

plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

#Random Initialization
def initialize_parameters_random(layers_dim):
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1, L):
      parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
      parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
      
    return parameters

parameters = initialize_parameters_random([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

parameters = model(train_X, train_Y, initialization = "random")
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

#He Initialization
def initialize_parameters_he(layers_dims):
  parameters = {}
  L = len(layers_dims)
  
  for l in range(1, L):
    parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
    parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    
  return parameters
  
parameters = initialize_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

parameters = model(train_X, train_Y, initialization = "he")
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


  
  
  
  
  
  
  
  
