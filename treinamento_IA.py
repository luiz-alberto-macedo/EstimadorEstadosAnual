import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
MasterFile = CurrentFolder / 'objs' / '13Bus' / 'Datasets' / 'dataset_anual_1000h_ProgBar_160s.xlsx'
data = pd.read_excel(MasterFile)

# Salvar as colunas de "hora" e "nome_barra" para adicionar na saída final
hora = data['hora']
nome_barra = data['nome_barra']
base = data['base']
geracao = data['Geracao']

# Selecionar as features e os targets
# Entradas: Potência injetada, tensão, fases conectadas
features = data[['base', 'Geracao',
                 'Fase_0', 'Fase_1', 'Fase_2',
                 'inj_pot_at_0', 'inj_pot_at_1', 'inj_pot_at_2', 
                 'inj_pot_rat_0', 'inj_pot_rat_1', 'inj_pot_rat_2',
                 'tensao_0', 'tensao_1', 'tensao_2']]

# Saídas: Ângulos de tensão estimados e tensões estimadas
targets = data[['ang_tensao_estimado_0', 'ang_tensao_estimado_1', 'ang_tensao_estimado_2',
                'tensao_estimada_0', 'tensao_estimada_1', 'tensao_estimada_2']]

# Tratar valores faltantes (opção: remover ou preencher)
features = features.fillna(0)
targets = targets.fillna(0)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Verificar o tamanho do conjunto de teste
print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")

# Normalizar os dados
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Definir a arquitetura do modelo DNN
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(y_train.shape[1]))  # Saída correspondente ao número de targets

# Compilar o modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Avaliar o modelo nos dados de teste
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test MAE: {mae}')

# Fazer previsões
predictions = model.predict(X_test)

# Inverter a normalização das previsões e dos valores reais
y_test_inversed = scaler_y.inverse_transform(y_test)
predictions_inversed = scaler_y.inverse_transform(predictions)

X_test_inversed = scaler_X.inverse_transform(X_test)

# Criar um DataFrame com as previsões
predictions_df = pd.DataFrame(predictions_inversed, columns=[
    'ang_tensao_estimado_0_pred', 'ang_tensao_estimado_1_pred', 'ang_tensao_estimado_2_pred',
    'tensao_estimada_0_pred', 'tensao_estimada_1_pred', 'tensao_estimada_2_pred'
])

# Restaurar o índice original do conjunto de teste
X_test_df = pd.DataFrame(X_test_inversed, columns=features.columns)
X_test_df = X_test_df.set_index(data.index[X_test_df.index])

# Recuperar o índice original do conjunto de teste
X_test_index = data.index[X_test_df.index]

# Concatenar as previsões ao DataFrame original das features (X_test) para manter a organização
result_df = pd.concat([X_test_df, predictions_df], axis=1)

# Adicionar as colunas 'hora' e 'nome_barra' ao DataFrame final
result_df = pd.concat([hora.reset_index(drop=True), nome_barra.reset_index(drop=True), result_df], axis=1)

# Concatenar as previsões ao DataFrame original das features (X_test) para manter a organização
result_df = pd.concat([hora.loc[X_test_index].reset_index(drop=True),
                       nome_barra.loc[X_test_index].reset_index(drop=True),
                       base.loc[X_test_index].reset_index(drop=True),
                       geracao.loc[X_test_index].reset_index(drop=True),
                       predictions_df], axis=1)


salvar_dataframe_como_csv(result_df)