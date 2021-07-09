from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from data_set import *
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime

sns.set(style="darkgrid")

arquivos = Mapper().arquivos

for arquivo in arquivos:

    entrada_treino = pd.read_csv('C:/Users/Matheus/Documents/GitHub/IA_EP1/data/' + arquivo['nome_problema'] + '.csv', header=None)
    ###########alterar para o caminho da pasta de dados para treino##########
    treinador = entrada_treino.drop(labels=63, axis=1)
    objetivo = np.squeeze(resultado[arquivo['nome_problema']])

    treinador_x, treinador_test, objetivo_y, objetivo_test = train_test_split(treinador, objetivo, test_size=7, stratify=objetivo)

    mlp = MLPClassifier(hidden_layer_sizes=63,
                              max_iter=10000,
                              alpha=1e-05,
                              activation='logistic',
                              solver='sgd',
                              learning_rate='adaptive',
                              learning_rate_init=0.6,
                              tol=1e-07,
                              verbose=True)

    MLP_fit = mlp.fit(treinador_x, objetivo_y)

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

    predictions = mlp.predict(treinador_test)
    print(f'ACURACIA: {accuracy_score(objetivo_test, predictions)}\n')

    def generate_loss_graph(mlp, graph_title):
        loss_curve = pd.DataFrame(mlp.loss_curve_)
        graph = sns.relplot(ci=None, kind="line", data=loss_curve)
        graph.fig.suptitle(graph_title)
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph.savefig(f"results\\graphs\\{graph_title}_{date}.png")

    generate_loss_graph(mlp, 'curva de erro X iteração')

    print(f'--- MATRIZ DE CONFUSAO ---\n{confusion_matrix(objetivo_test.argmax(axis=1), predictions.argmax(axis=1))}\n')