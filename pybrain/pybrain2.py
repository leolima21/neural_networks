# imports
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer

# configracao da rede neural caso o problema n seja linear
'''rede = buildNetwork(2, 3, 1, outclass = SoftmaxLayer,
                    hiddenclass = SigmoidLayer, bias = False)
print(rede['in'])
print(rede['hidden0'])
print(rede['out'])
print(rede['bias'])'''

# criacao da rede neural com 2 neuronios na entrada, 3 na oculta e 1 na saida
rede = buildNetwork(2, 3, 1)

# preparacao de uma base de dados
base = SupervisedDataSet(2, 1)
base.addSample((0, 0), (0, ))
base.addSample((0, 1), (1, ))
base.addSample((1, 0), (1, ))
base.addSample((1, 1), (0, ))

# print da entrada e da classe
print('INPUT: ')
print(base['input'])
print('TARGET: ')
print(base['target'])

# criacao do objeto de treinamento 
treinamento = BackpropTrainer(rede, dataset = base, learningrate = 0.01,
                              momentum = 0.06)

# treinamento com 30 mil epocas
for i in range(1, 30000):
    erro = treinamento.train()
    if i % 1000 == 0:
        print("Erro: %s" % erro)

# print de uma classificacao apos treinamento 
print(' ') 
print('RESULTADO DA CLASSIFICACAO DE ENTRADA XOR: ')        
print(rede.activate([0, 0]))
print(rede.activate([1, 0]))
print(rede.activate([0, 1]))
print(rede.activate([1, 1]))