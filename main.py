"""
    Classe responsavel por por controlar toda a execução do algoritmo Multi-layer Perceptron, tanto para sua fase
    treinamento, quanto para sua fase de teste
    Classe que implementa o algoritmo Multi-layer Perceptron (MLP).
"""
from sklearn.neural_network import MLPClassifier #implementa o MLP
from sklearn.model_selection import train_test_split #define os conjuntos de treino e teste
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #metricas
from data_set import *
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
np.set_printoptions(threshold=sys.maxsize)
#configuar parametros para geração de gráficos

arquivos = Mapper().arquivos


for arquivo in arquivos:

    entrada_treino = pd.read_csv('C:/Users/Matheus/Documents/GitHub/IA_EP1/data/' + arquivo['nome_problema'] + '.csv', header=None) #alterar para o caminho da pasta de dados para treino
    treinador = entrada_treino.drop(labels=63, axis=1)
    #DÚVIDA
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
    #CRIA O MLP

    MLP_fit = mlp.fit(treinador_x, objetivo_y)#altera
    #ENCAIXA O MODELO DE DADOS x COM O ALVO y

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
    ###é uma lista de matrizes de peso, em que a matriz de peso no índice i representa os pesos entre a camada i e a camada i + 1.
    print(f'Pesos Camada de Entrada: \n{mlp.coefs_[0]}')
    print(f'Pesos Camada de Saida: \n{mlp.coefs_[1]}')
    ###é uma lista de vetores de bias, em que o vetor no índice i representa os valores de bias adicionados à camada i + 1.
    print(f'Bias Camada de Entrada: \n{mlp.intercepts_[0]}')
    print(f'Bias Camada de Saida: \n{mlp.intercepts_[1]}\n')

    print('METRICAS')
    predictions_proba = mlp.predict_proba(treinador_test)
    predictions = mlp.predict(treinador_test)
    print(f'ACURACIA: {accuracy_score(objetivo_test, predictions)}')

        ##curva de erro x iteracao
        # print('--- ERRO X ITERACAO ---\nCurva do erro calculado em funcao da perda x iteracao.\n')
        # loss_curve = pd.DataFrame(mlp.loss_curve_)
        # graph = sns.relplot(ci=None, kind="line", data=loss_curve)
        # graph
        # sys.stdout.close()
        # gg(loss_curve, aes(x='iterations', y='loss')) + gg.geom_line()

    ## ESTIMADOR UTILIZADO PARA DEFINIR A MELHOR CONFIGURACAO DA REDE
    # ###define os parametros que serao combinados pelo GridSearch
    # parameter_space = {
    #     'hidden_layer_sizes': [15, 20, 45, 63, 70],
    #     'activation': ['relu', 'tanh', 'logistic'],
    #     'solver': ['sgd', 'adam'],
    #     'alpha': [0.00001, 0.0001, 0.001],
    #     'learning_rate': ['constant' ,'adaptive'],
    #     'learning_rate_init': [0.25, 0.5, 0.6, 0.8, 0.7],
    #     'tol': [0.00001, 0.000001, 0.0001, 0.001, 0.01, 0.0000001]
    # }
    #
    #mlp = MLPClassifier(max_iter = 10000)
    #
    # ##chamada que implementa o GridSearch
    # clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=7)
    #
    # ##realiza o fit com para as combinacoes retornadas pelo GridSearch utilizando os parametros pre-definidos
    # clf.fit(treinador_x, objetivo_y)
    #
    # ##log GridSearch
    # print('Best estimator found:\n', clf.best_estimator_)
    # print('Best parameters found:\n', clf.best_params_)
    # print('Best index found:\n', clf.best_index_)
    # print('Best score index found:\n', clf.best_score_)
    # print('CV:\n', clf.cv)
    # print('CV Result:\n', clf.cv_results_)
    # print('Predict:\n', clf.predict)
    # print('Error score:\n', clf.error_score)
    # print('Param grid:\n', clf.param_grid)
    # print('Multimetric:\n', clf.multimetric_)

    ### METRICAS PARTE3
    ##matriz de confusao
    # print(f'--- MATRIZ DE CONFUSAO ---\n{confusion_matrix(objetivo_test.argmax(axis=1), predictions.argmax(axis=1))}\n')

    ##classificador
    # print(
    #     f'--- OUTRAS METRICAS DO CLASSIFICADOR ---\n{classification_report(objetivo_test.argmax(axis=1), predictions.argmax(axis=1))}\n')

##realiza a leitura do csv para pegar os dados para o MLP
# treinador = pd.read_csv('../inputs/Part-1/caracteres-limpos.csv', header=None)
# treinador = treinador.drop(labels=63, axis=1)