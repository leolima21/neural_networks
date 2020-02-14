import numpy as np

class Perceptron:   
  def __init__(self, inp, w):
    self.input = inp
    self.weight = w

  def soma(self):
    return self.input.dot(self.weight)

  def stepFunction(self, soma):
    if (soma >= 1):
      return 1
    return 0

if __name__ == '__main__':
  entradas = np.array([1, 7, 5])
  pesos = np.array([0.8, 0.1, 0])

  percept = Perceptron(entradas, pesos)
  print(percept.soma())
  print(percept.stepFunction(percept.soma()))