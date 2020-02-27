# imports
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

# criacao de uma rede do tipo feed forward
rede = FeedForwardNetwork()

# criação de cada uma das camadas com a quantidade de neuronios desejada
camadaEntrada = LinearLayer(2)
camadaOculta = SigmoidLayer(3)
camadaSaida = SigmoidLayer(1)
# unidade de bias para a camada oculta e a camada de saida
bias1 = BiasUnit()
bias2 = BiasUnit()

# adicionar as camadas criadas na rede
rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

# ligacao entre as camadas
entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias1, camadaOculta)
biasSaida = FullConnection(bias2, camadaSaida)

# construcao de fato da rede neural
rede.sortModules()

# prints
print('REDE: ')
print(rede)
print('CAMADA ENTRADA-OCULTA PARAMETROS: ')
print(entradaOculta.params)
print('CAMADA OCULTA-SAIDA PARAMETROS: ')
print(ocultaSaida.params)
print('CAMADA BIAS-OCULTA PARAMETROS: ')
print(biasOculta.params)
print('CAMADA BIAS-SAIDA PARAMETROS: ')
print(biasSaida.params)


