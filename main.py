#Gustavo Sabino nºUSP10723732
#Matheus Amorim nºUSP10882810
#Vinícius Zacarias nºUSP10783198

# Arquivo responsável pela implementação da rede neural Multilayer Perceptron (MLP).
# Neste arquivo estão implementadas as fases de treinamento e teste do MLP.

from sklearn.neural_network import MLPClassifier #Função que implementa o MLP
from sklearn.model_selection import train_test_split #Função que gerencia as matrizes de entrada e a separa.
from sklearn.metrics import accuracy_score, confusion_matrix #Funções de métrica para análise do MLP
from data_set import *
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime

sns.set(style="darkgrid") #Estilo do gráfico

arquivos = Mapper().arquivos

for arquivo in arquivos:

    entrada_treino = pd.read_csv('C:/Users/Matheus/Documents/GitHub/IA_EP1/data/' + arquivo['nome_problema'] + '.csv', header=None) #Busca a matriz de caracteres
    ###########alterar para o caminho da pasta de dados para treino##########
    treinador = entrada_treino.drop(labels=63, axis=1)
    objetivo = np.squeeze(resultado[arquivo['nome_problema']])

    treinador_x, treinador_test, objetivo_y, objetivo_test = train_test_split(treinador, objetivo, test_size=7, stratify=objetivo) #Gerencia a entrada e a distribui para treinamento e teste

    mlp = MLPClassifier(hidden_layer_sizes=63,
                              max_iter=10000,
                              alpha=1e-05,
                              activation='logistic',
                              solver='sgd',
                              learning_rate='adaptive',
                              learning_rate_init=0.6,
                              tol=1e-07,
                              verbose=True)
    #Exceuta o treinamento do MLP seguindo as métricas passadas

    MLP_fit = mlp.fit(treinador_x, objetivo_y)
    #Ajusta o modelo à matriz de dados X e alvo (s) y

    ######Saídas para análise da execução######

    print()
    print('PARAMETROS DE INICIALIZACAO DA REDE \n NUMERO DE NEURONIOS \n')
    print(f'Camada de Entrada: 63')
    print(f'Camada Escondida: {mlp.hidden_layer_sizes}')
    print(f'Camada de Saida: {mlp.n_outputs_}\n')

    print('PARAMETROS DE CONFIGURACAO DA REDE')
    print(f'Numero de Epocas: {mlp.n_iter_}')
    print(f'Funcao de Ativacao: {mlp.activation}')
    print(f'Solver utilizado: {mlp.solver}')
    print(f'Taxa de Aprendizado: {mlp.learning_rate}')
    print(f'Taxa de Aprendizado Inicial: {mlp.learning_rate_init}')
    print(f'Tolerancia: {mlp.tol}')
    print(f'Penalidade: {mlp.alpha}\n')

    print('PARAMETROS FINAIS DA REDE')
    print(f'Pesos Camada de Entrada: \n{mlp.coefs_[0]}')
    print(f'Pesos Camada de Saida: \n{mlp.coefs_[1]}')
    print(f'Bias Camada de Entrada: \n{mlp.intercepts_[0]}')
    print(f'Bias Camada de Saida: \n{mlp.intercepts_[1]}\n')

    ####Fase de teste####
    predictions = mlp.predict(treinador_test)
    print(f'ACURACIA: {accuracy_score(objetivo_test, predictions)}\n')


    def generate_loss_graph(mlp, graph_title): #Define como deve ser gerada a curva de erro.
        loss_curve = pd.DataFrame(mlp.loss_curve_)
        graph = sns.relplot(ci=None, kind="line", data=loss_curve)
        graph.fig.suptitle(graph_title)
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph.savefig(f"results\\graphs\\{graph_title}_{date}.png")

    #Gera o gráfico de perda 'Erro x Iteração'.
    generate_loss_graph(mlp, 'curva de erro X iteração')

    #Gera a matriz de confusão.
    print(f'--- MATRIZ DE CONFUSAO ---\n{confusion_matrix(objetivo_test.argmax(axis=1), predictions.argmax(axis=1))}\n')