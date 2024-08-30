#import tensorflow as tf
#from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from pathlib import Path
import time

def salvar_dataframe_como_csv(df_formatado):
    # Salvar o DataFrame em um arquivo CSV
    path = Path(__file__)
    CurrentFolder = path.parent
    MasterFile = CurrentFolder / 'objs' / '13Bus' / 'Previsoes' / 'resultados_IA.csv'

    # Criar os diretórios, se necessário
    MasterFile.parent.mkdir(parents=True, exist_ok=True)

    df_formatado.to_csv(MasterFile, index=False)

# Carregar o dataset
path = Path(__file__)
CurrentFolder = path.parent
MasterFile = CurrentFolder / 'objs' / '13Bus' / 'Datasets' / 'dataset_anual_4380h.csv'
data = pd.read_csv(MasterFile)

# Salvar as colunas de "hora" e "nome_barra" para adicionar na saída final
hora = data['hora']
nome_barra = data['nome_barra']
base = data['base']
geracao = data['Geracao']

# Selecionar as features e os targets
# Entradas: Potência injetada, tensão, fases conectadas
features = data[['Geracao',
                 'Fase_0', 'Fase_1', 'Fase_2',
                 'inj_pot_at_0', 'inj_pot_at_1', 'inj_pot_at_2', 
                 'inj_pot_rat_0', 'inj_pot_rat_1', 'inj_pot_rat_2',
                 'tensao_0', 'tensao_1', 'tensao_2']]

# Saídas: Ângulos de tensão estimados e tensões estimadas
targets = data[['ang_tensao_estimado_0', 'ang_tensao_estimado_1', 'ang_tensao_estimado_2',
                'tensao_estimada_0', 'tensao_estimada_1', 'tensao_estimada_2']]


# Tratar valores faltantes (opção: remover ou preencher)
X = features.fillna(value=0)
y = targets.fillna(value=0)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizando os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criando e treinando o modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Fazendo previsões
y_pred = model.predict(X_test_scaled)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Fazendo previsões para todo o dataset
X_all_scaled = scaler.transform(X)
y_all_pred = model.predict(X_all_scaled)

# Adicionando as previsões ao dataframe original
for i, target in enumerate(targets):
    data[f'{target}_pred'] = y_all_pred[:, i]

salvar_dataframe_como_csv(data)