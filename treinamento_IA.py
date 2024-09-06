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
    MasterFile = CurrentFolder / 'objs' / '13Bus' / 'Previsoes' / 'resultados_IA_V2.csv'

    # Criar os diretórios, se necessário
    MasterFile.parent.mkdir(parents=True, exist_ok=True)

    df_formatado.to_csv(MasterFile, index=False)

# Carregar o dataset
path = Path(__file__)
CurrentFolder = path.parent
MasterFile = CurrentFolder / 'objs' / '13Bus' / 'Datasets' / 'dataset_anual_8760h_11h.csv'
data = pd.read_csv(MasterFile)

# Salvar as colunas de "hora" e "nome_barra" para adicionar na saída final
hora = data['hora']
nome_barra = data['nome_barra']
base = data['base']
geracao = data['Geracao']

metade1 = data.iloc[:112128,:] #80% das 140160 linhas do dataframe
metade2 = data.iloc[112128:,:]

'''
print(metade1)
print("-----------")
print(metade2)
'''

# Selecionar as features e os targets
# Entradas: Potência injetada, tensão, fases conectadas
features_metade1 = metade1[['Geracao',
                 'Fase_0', 'Fase_1', 'Fase_2',
                 'inj_pot_at_0', 'inj_pot_at_1', 'inj_pot_at_2', 
                 'inj_pot_rat_0', 'inj_pot_rat_1', 'inj_pot_rat_2',
                 'tensao_0', 'tensao_1', 'tensao_2']]

# Saídas: Ângulos de tensão estimados e tensões estimadas
targets_metade1 = metade1[['ang_tensao_estimado_0', 'ang_tensao_estimado_1', 'ang_tensao_estimado_2',
                'tensao_estimada_0', 'tensao_estimada_1', 'tensao_estimada_2']]


# Tratar valores faltantes (opção: remover ou preencher)
Entrada_metade1 = features_metade1.fillna(value=0)
Saida_metade1 = targets_metade1.fillna(value=0)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(Entrada_metade1, Saida_metade1, test_size=0.2, random_state=42)

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

'''
# Fazendo previsões para todo o dataset
X_scalled_metade1 = scaler.transform(Entrada_metade1)
y_pred_metade1 = model.predict(X_scalled_metade1)
'''

# Fazendo previsões para segunda metade do dataset
features_metade2 = metade2[['Geracao',
                 'Fase_0', 'Fase_1', 'Fase_2',
                 'inj_pot_at_0', 'inj_pot_at_1', 'inj_pot_at_2', 
                 'inj_pot_rat_0', 'inj_pot_rat_1', 'inj_pot_rat_2',
                 'tensao_0', 'tensao_1', 'tensao_2']]

# Saídas: Ângulos de tensão estimados e tensões estimadas
targets_metade2 = metade2[['ang_tensao_estimado_0', 'ang_tensao_estimado_1', 'ang_tensao_estimado_2',
                'tensao_estimada_0', 'tensao_estimada_1', 'tensao_estimada_2']]


# Tratar valores faltantes (opção: remover ou preencher)
Entrada_metade2 = features_metade2.fillna(value=0)

# Fazendo previsões para segunda metade do dataset
X_scalled_metade2 = scaler.transform(Entrada_metade2)
y_pred_metade2 = model.predict(X_scalled_metade2)

# Adicionando as previsões ao dataframe original
for i, target in enumerate(targets_metade2):
    metade2[f'{target}_pred'] = y_pred_metade2[:, i]

salvar_dataframe_como_csv(metade2)
