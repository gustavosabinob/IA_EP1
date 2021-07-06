#classe responsável pela implementação, treinamento e teste do algoritmo MLP

from sklearn.neural_network import MLPClassifier

# X = [[0., 0.], [1., 1.]]
# y = [0, 1]
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(X, y)
# MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1, solver='lbfgs')

"""
    Classe responsavel por por controlar toda a execução do algoritmo Multi-layer Perceptron, tanto para sua fase
    treinamento, quanto para sua fase de teste
"""
from sklearn.model_selection import train_test_split  # define os conjuntos de treino e teste
import sklearn.metrics as metrics

import sys
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")
np.set_printoptions(threshold=sys.maxsize)

start = datetime.datetime.now()

train_df = pd.read_csv('folds\\caracteres-limpo.csv', header=None)
test_df = pd.read_csv('folds\\caracteres-ruido.csv', header=None)
test2_df = pd.read_csv('folds\\caracteres_ruido20.csv', header=None)
#train_targets = pd.read_csv('folds\\train_targets.csv', header=None)
#test_targets = pd.read_csv('folds\\test_targets.csv', header=None)

def init_mlp(neuronio, taxa, epoca):
    # taxa de aprendizado adaptativa, talvez tenha que mudar porconta da estrategia em grade
    config_name = f"{neuronio}_{taxa}_{epoca}"
    #sys.stdout = open(f"results\\{config_name}", "w")

    mlp_model = MLPClassifier(hidden_layer_sizes=neuronio,
                              max_iter=epoca,
                              alpha=1e-05,
                              activation='logistic',
                              solver='sgd',
                              learning_rate='adaptive',
                              learning_rate_init=taxa)
    #mlp_model.fit(train_df,train_df)

    print('PARAMETROS DE INICIALIZACAO DA REDE\nNUMERO DE NEURONIOS ')
    print(f'Camada de Entrada: ',neuronio)
    print(f'Camada Escondida: {mlp_model.hidden_layer_sizes}')
    print(f'Camada de Saida: {mlp_model.n_outputs_}\n')

    print('--- PARAMETROS DE CONFIGURACAO DA REDE ---')
    print(f'Numero de Epocas: {mlp_model.n_iter_}')
    print(f'Taxa de Aprendizado: {mlp_model.learning_rate}')
    print(f'Taxa de Aprendizado Inicial: {mlp_model.learning_rate_init}')

    print('METRICAS\n')
    predictions = mlp_model.predict(test_df)
    print("RESULTADOS:\n")
    print(predictions)

    return mlp_model


def generate_loss_graph(mlp, graph_title):
    loss_curve = pd.DataFrame(mlp.loss_curve_)
    graph = sns.relplot(ci=None, kind="line", data=loss_curve)
    graph.fig.suptitle(graph_title)
    graph.savefig(f"results\\graphs\\{graph_title}.png")


def generate_final_graphs(predictions):
    # Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(metrics.confusion_matrix(test_df, predictions), annot=True,
                ax=ax, fmt='d', cmap='Reds')
    ax.set_title("Matriz de Confusão", fontsize=18)
    ax.set_ylabel("Rótulo Verdadeiro")
    ax.set_xlabel("Rótulo Previsto")
    plt.tight_layout()
    plt.savefig(f"results\\graphs\\best_mlp_confusion_matrix.png")

    # Precision x Recall
    precisions, recalls, thresholds = metrics.precision_recall_curve(test_df, predictions)
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precisão')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Revocação')
    plt.xlabel('Limiar de Decisão')
    plt.legend(loc='center right')
    plt.ylim([0, 1])
    plt.title('Precisão x Revocação', fontsize=14)
    plt.savefig(f"results\\graphs\\best_mlp_precision_recall.png")

    #  ROC
    fpr, tpr, thresholds = metrics.roc_curve(test_df, predictions)
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.plot(fpr, tpr, linewidth=2, label='Regressão Logística')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.legend(loc='lower right')
    plt.title('Curva ROC', fontsize=14)
    plt.savefig(f"results\\graphs\\best_mlp_roc_curve.png")

    # Acurácia, comparativo com a acuracia do classification_report
    accuracy = metrics.accuracy_score(test_df, predictions)
    print(f'ACURACIA: {accuracy}\n')

    report = metrics.classification_report(test_df, best_mlp_predictions)
    print(f'Relatório de Classificação\n{report}\n')


# generate_csv()

# estrategia baseada em grade para testar a rede MLP
max_accuracy = 0
best_config = []
best_mlp = None

neuronios = [21]
taxas = [0.3]
epocas = [300]

for neuronio in neuronios:
    for taxa in taxas:
        for epoca in epocas:
            mlp_model = init_mlp(neuronio, taxa, epoca)
            predictions = mlp_model.predict(test_df)
            accuracy = metrics.accuracy_score(test_df, predictions)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_config.extend([neuronio, taxa, epoca])
                best_mlp = mlp_model

graph_title = f"melhor_mlp_gráfico_erro_{best_config[0]}_{best_config[1]}_{best_config[2]}"
#generate_loss_graph(best_mlp, graph_title)
best_mlp_predictions = best_mlp.predict(test_df)

# abaixo executamos o MLP com a estratégia de early stopping para a melhor
# configuração encontrada entre cada combinação de hiperparâmetros
best_mlp_early_stopping = init_mlp(best_config[0], best_config[1], best_config[2], True)
es_graph_title = f"es_melhor_mlp_gráfico_erro_{best_config[0]}_{best_config[1]}_{best_config[2]}"
#generate_loss_graph(best_mlp_early_stopping, es_graph_title)

def NEW_generate_loss_graph(mlp, mlp_es):
    loss_curve_best = pd.DataFrame(mlp.loss_curve_)
    loss_curve_best_es = pd.DataFrame(mlp_es.loss_curve_)

    plt.plot(loss_curve_best, 'g-', label='Treinamento')
    plt.plot(loss_curve_best_es, 'b--', label='Validação')
    plt.xlabel('Número de Épocas')
    plt.legend(loc='center right')
    plt.title('Treinamento x Validação', fontsize=14)
    plt.savefig(f"results\\graphs\\train_validation_error.png")

NEW_generate_loss_graph(best_mlp, best_mlp_early_stopping)



# os gráficos são gerados para comparar a função do erro entre os dois modelos

# report = metrics.classification_report(test_targets, best_mlp_predictions)
# print(f'Relatório de Classificação\n{report}\n')

generate_final_graphs(best_mlp_predictions)

end = datetime.datetime.now()
runtime = start - end