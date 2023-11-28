from sklearn.base import BaseEstimator, TransformerMixin

# classes são utilizadas como base para a criação de estimadores personalizados (modelos) 
# e transformadores de dados.
# BaseEstimator: fornece implementações padrão para os métodos get_params() e set_params()
# TransformerMixin inclui um método adicional chamado fit_transform(), que combina os métodos fit() 
# e transform() em um único passo eficiente.

from sklearn.preprocessing import MinMaxScaler



# Importa as classes BaseEstimator e TransformerMixin do módulo sklearn.base
class MinMax(BaseEstimator, TransformerMixin):

    # O método __init__ é chamado quando uma instância da classe é criada
    def __init__(self, min_max_scaler=['Devedor', 
                                       'TaxaDesemprego', 
                                       'MensalidadesEmDia', 
                                       'Bolsista', 
                                       'UnidadesCurriculares1SemestreAprovado', 
                                       'UnidadesCurriculares1SemestreAvaliacoes', 
                                       'UnidadesCurriculares1SemestreGrau']):
        # Inicializa a instância com uma lista de colunas a serem aplicadas à transformação MinMaxScaler
        self.min_max_scaler = min_max_scaler

    # O método fit é utilizado para treinar o transformador
    def fit(self, df):
        # Neste caso, não é necessário realizar nenhum treinamento, então apenas retorna a própria instância
        return self

    # O método transform aplica a transformação MinMaxScaler às colunas especificadas
    def transform(self, df):
        # Cria uma instância de MinMaxScaler
        min_max_enc = MinMaxScaler()

        # Aplica a transformação MinMaxScaler às colunas especificadas no DataFrame
        df[self.min_max_scaler] = min_max_enc.fit_transform(df[self.min_max_scaler])

        # Retorna o DataFrame modificado
        return df
