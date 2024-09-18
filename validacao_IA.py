from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from pathlib import Path
import time
import joblib

def salvar_dataframe_como_csv(df_formatado, Loadshape):
    # Salvar o DataFrame em um arquivo CSV
    path = Path(__file__)
    CurrentFolder = path.parent
    MasterFile = CurrentFolder / 'objs' / '13Bus' / 'Previsoes' / f'{Loadshape}' / 'resultados_IA_V4.csv'

    # Criar os diretórios, se necessário
    MasterFile.parent.mkdir(parents=True, exist_ok=True)

    df_formatado.to_csv(MasterFile, index=False)

# Carregar o dataset
path = Path(__file__)
CurrentFolder = path.parent
Loadshape = 'Loadshape2'
MasterFile = CurrentFolder / 'objs' / '13Bus' / 'Datasets' / f'{Loadshape}' / 'dataset_anual_0h_8760h_8760h.csv'
data = pd.read_csv(MasterFile)

# Carregar o modelo salvo para testar com outro dataset
model_filename = CurrentFolder / 'objs' / '13Bus' / 'IAs_treinadas' / "Modelo_treinado_Loadshape1_Misturado_rf.pkl"
modelo_carregado = joblib.load(model_filename)
print("Modelo carregado com sucesso!")

# Carregar o scaler salvo
scaler_filename = CurrentFolder / 'objs' / '13Bus' / 'IAs_treinadas' / "scaler_Loadshape1_Misturado.pkl"
scaler = joblib.load(scaler_filename)
print("Scaler carregado com sucesso!")

# Salvar as colunas de "hora" e "nome_barra" para adicionar na saída final
hora = data['hora']
nome_barra = data['nome_barra']
base = data['base']
geracao = data['Geracao']

# Fazendo previsões para segunda metade do dataset
features = data[['Geracao',
                 'Fase_0', 'Fase_1', 'Fase_2',
                 'inj_pot_at_0', 'inj_pot_at_1', 'inj_pot_at_2', 
                 'inj_pot_rat_0', 'inj_pot_rat_1', 'inj_pot_rat_2',
                 'tensao_0', 'tensao_1', 'tensao_2']]

# Saídas: Ângulos de tensão estimados e tensões estimadas
targets = data[['ang_tensao_estimado_0', 'ang_tensao_estimado_1', 'ang_tensao_estimado_2',
                'tensao_estimada_0', 'tensao_estimada_1', 'tensao_estimada_2']]


# Tratar valores faltantes (opção: remover ou preencher)
Entrada = features.fillna(value=0)

# Fazendo previsões para segunda metade do dataset
#scaler = StandardScaler()
#model = RandomForestRegressor(n_estimators=100, random_state=42)

inicio_validacao = time.time()
X_scalled = scaler.transform(Entrada)
y_pred = modelo_carregado.predict(X_scalled)
fim_validacao = time.time()

print(f'Previsões da IA realizadas em {fim_validacao-inicio_validacao}s')

# Adicionando as previsões ao dataframe original
for i, target in enumerate(targets):
    data[f'{target}_pred'] = y_pred[:, i]

salvar_dataframe_como_csv(data, Loadshape)