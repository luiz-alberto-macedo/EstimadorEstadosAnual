from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from pathlib import Path
import time
import joblib

def salvar_dataframe_como_csv(df_formatado):
    # Salvar o DataFrame em um arquivo CSV
    path = Path(__file__)
    CurrentFolder = path.parent
    MasterFile = CurrentFolder / 'objs' / '13Bus' / 'Previsoes' / 'Loadshape_1_2' / 'resultados_IA_V5.csv'

    # Criar os diretórios, se necessário
    MasterFile.parent.mkdir(parents=True, exist_ok=True)

    df_formatado.to_csv(MasterFile, index=False)

# Função para adicionar 5 horas ao valor da coluna 'hora'
def adicionar_horas(hora_str, valor_a_ser_somado):
    # Extrai o número da string 'hora_x'
    hora_numero = int(hora_str.split('_')[1])
    # Soma 5 horas
    nova_hora_numero = hora_numero + valor_a_ser_somado
    # Retorna o novo valor no formato 'hora_x'
    return f'hora_{nova_hora_numero}'

# Carregar os datasets
path = Path(__file__)
CurrentFolder = path.parent
MasterFile_Loadshape1 = CurrentFolder / 'objs' / '13Bus' / 'Datasets' / 'Loadshape1' / 'dataset_anual_8760h_11h.csv'
MasterFile_Loadshape2 = CurrentFolder / 'objs' / '13Bus' / 'Datasets' / 'Loadshape2' / 'dataset_anual_0h_8760h_8760h.csv'
data_Loadshape1 = pd.read_csv(MasterFile_Loadshape1)
data_Loadshape2 = pd.read_csv(MasterFile_Loadshape2)

# Modifica a coluna 'hora' adicionando 8760h à coluna "hora_x"
data_Loadshape2['hora'] = data_Loadshape2['hora'].apply(lambda x: adicionar_horas(x, 8760))

datasets_concatenados = pd.concat([data_Loadshape1, data_Loadshape2], ignore_index=True)

# Obtendo as horas únicas
horas_unicas = datasets_concatenados['hora'].unique()

# Embaralhando as horas
horas_embaralhadas = np.random.permutation(horas_unicas)

# Criando um dicionário de mapeamento das horas originais para as embaralhadas
mapeamento_horas = dict(zip(horas_unicas, horas_embaralhadas))

# Aplicando o mapeamento para criar uma nova coluna de horas embaralhadas
datasets_concatenados['hora_embaralhada'] = datasets_concatenados['hora'].map(mapeamento_horas)

# Reordenando o DataFrame com base nas horas embaralhadas
datasets_concatenados_embaralhados = datasets_concatenados.sort_values(['hora_embaralhada','nome_barra'])

# Removendo a coluna auxiliar de horas embaralhadas
datasets_concatenados_embaralhados = datasets_concatenados_embaralhados.drop('hora_embaralhada', axis=1)

# Salvar as colunas de "hora" e "nome_barra" para adicionar na saída final
hora = datasets_concatenados_embaralhados['hora']
nome_barra = datasets_concatenados_embaralhados['nome_barra']
base = datasets_concatenados_embaralhados['base']
geracao = datasets_concatenados_embaralhados['Geracao']

porcentagem_para_treino = int(len(datasets_concatenados_embaralhados.index)*0.5) #50% das 280320 linhas do dataframe


metade1 = datasets_concatenados_embaralhados.iloc[:porcentagem_para_treino,:] 
metade2 = datasets_concatenados_embaralhados.iloc[porcentagem_para_treino:,:]

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
inicio_treinamento = time.time()
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
fim_treinamento = time.time()

print(f'IA treinada em {fim_treinamento-inicio_treinamento}s')

# Salvar o modelo treinado
model_filename = CurrentFolder / 'objs' / '13Bus' / 'IAs_treinadas' / "Modelo_treinado_Loadshape1_Misturado_rf.pkl"
joblib.dump(model, model_filename)
print(f"Modelo salvo como {model_filename}")

# Salvar o scaler ajustado
scaler_filename = CurrentFolder / 'objs' / '13Bus' / 'IAs_treinadas' / "scaler_Loadshape1_Misturado.pkl"
joblib.dump(scaler, scaler_filename)
print(f"Scaler salvo em: {scaler_filename}")

# Fazendo previsões
y_pred = model.predict(X_test_scaled)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

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
