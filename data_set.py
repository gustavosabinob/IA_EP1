"""
    Classe responsavel por fazer o mapeamento dos arquivos .csv para teste e para treino da rede
    tratando as respostas esperadas para o problema e gerando dicionarios para melhor manipulá-los
    Functions:
        init(self): Inicia a leitura do arquivo.
        handle_input: Funcao que le o arquivo .csv retornando um dicioncario de dados para ser usado na rede neural.
        arquivo(self): Funcao que guarda o objeto do retorno da funcao 'handle_input' na variavel _arquivo.
        arquivo(self,value): Funcao que 'seta' os valores do objeto guardado pela funcao 'arquivo(self)' na variavel _arquivo.
        get_target(self, target): Funcao que retorna o valor esperado dentro da rede de acordo com o target alfanumerico
"""

import csv

# Abaixo nomeamos os arquivos que são utilizados para treinar a rede neural
conjunto_teste = [#'problemAND.csv', 'problemOR.csv', 'problemXOR.csv',
                        'caracteres-limpos.csv', 'caracteres-ruidos.csv']
# ARQUIVOS_PARA_TREINO = ['caracteres-limpos.csv']

# Abaixo nomeamos os arquivos que são utilizados para testar a rede neural
conjunto_treino = ['caracteres-ruidos.csv']

resultado = {
    'caracteres-ruidos': [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ],
    'caracteres-limpos': [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ]
}

class Mapper:
    def __init__(self):
        self._arquivos = self.get_arquivos()
        self.arquivos_teste = self.get_arquivo_de_teste()

    @property
    def arquivos(self):
        return self._arquivos

    @arquivos.setter
    def arquivos(self, value):
        self._arquivos = value

    def get_arquivos(self):
        resultado = []
        for arquivo in conjunto_treino:
            resultado.append(self.input(arquivo))
        return resultado

    def get_arquivo_de_teste(self):
        resultado = []
        for arquivo in conjunto_teste:
            resultado.append(self.input(arquivo))
        return resultado

    def input(self, filename):
        inputs = []
        caminho_arquivo = 'C:/Users/Matheus/Documents/GitHub/IA_EP1/data/' + filename #alterar para o caminho da pasta de dados para treino
        with open(caminho_arquivo, 'rt', encoding="utf-8-sig") as data:
            dados_arquivo = csv.reader(data)

            for linha in dados_arquivo:
                target = self.get_alvo(linha[-1])
                sample = linha[:-1]
                inputs.append({
                    'target_description': linha[-1],
                    'target': target,
                    'sample': sample
                })

            resultado = {'nome_problema': filename[:-4],
                      'inputs': inputs}
        return resultado

    def get_alvo(self, target):
        dict = {
            'A': [1, 0, 0, 0, 0, 0, 0],
            'B': [0, 1, 0, 0, 0, 0, 0],
            'C': [0, 0, 1, 0, 0, 0, 0],
            'D': [0, 0, 0, 1, 0, 0, 0],
            'E': [0, 0, 0, 0, 1, 0, 0],
            'J': [0, 0, 0, 0, 0, 1, 0],
            'K': [0, 0, 0, 0, 0, 0, 1],
            '0': [0],
            '1': [1]
        }

        """
             O target dos outros problemas e transformado em lista para que a rede seja
            generica para todos os problemas em questao
        """

        return dict[target]