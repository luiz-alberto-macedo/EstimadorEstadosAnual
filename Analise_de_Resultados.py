import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.dates as mdates
import seaborn as sns
import re

# Função para calcular métricas
def calcular_metricas(real, estimada):
    mse = mean_squared_error(real, estimada)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real, estimada)
    corr, _ = pearsonr(real, estimada)
    return rmse, mae, corr

# Função para exibir métricas
def exibir_metricas(ax, title, metrics, pos):
    text = '\n'.join([f'{name}: {value:.6f}' for name, value in metrics.items()])
    ax.text(pos[0], pos[1], text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(title)

# Função para plotar tensões
def plotar_tensoes(barra, fase, ax):
    ax.plot(barra['hora'], barra[f'tensao_{fase}'], label='Tensão Medidor')
    ax.plot(barra['hora'], barra[f'tensao_estimada_{fase}'], label='Tensão Estimada', linestyle='--')
    ax.plot(barra['hora'], barra[f'tensao_estimada_{fase}_pred'], label='Tensão Prevista IA', linestyle='--')

def calcular_erros(data, fases):
    erros = {}
    for fase in fases:
        erros[f'ia_{fase}'] = np.abs(data[f'tensao_{fase}'] - data[f'tensao_estimada_{fase}_pred'])
        erros[f'estimada_{fase}'] = np.abs(data[f'tensao_{fase}'] - data[f'tensao_estimada_{fase}'])
        erros[f'ia_estimador_{fase}'] = np.abs(data[f'tensao_estimada_{fase}'] - data[f'tensao_estimada_{fase}_pred'])
    return erros

def plotar_distribuicao_erros(erros, fases):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=True, sharex=True)
    cores = {'ia': ['blue', 'green', 'orange'], 'estimada': ['yellow', 'red', 'indigo']}
    
    for i, fase in enumerate(fases):
        # Comparação IA vs Medidor
        sns.histplot(erros[f'ia_{fase}'], bins=30, label=f'Erro IA Fase {fase}', color=cores['ia'][i], kde=True, ax=axs[0, i])
        sns.histplot(erros[f'estimada_{fase}'], bins=30, label=f'Erro Estimador de Estados Fase {fase}', color=cores['estimada'][i], kde=True, ax=axs[0, i])
        axs[0, i].set_title(f'Distribuição dos Erros em relação ao medidor - Fase {fase}')
        axs[0, i].set_xlabel('Erro Absoluto (p.u.)')
        axs[0, i].set_ylabel('Frequência')
        axs[0, i].legend()

        # Comparação IA vs EE
        sns.histplot(erros[f'ia_estimador_{fase}'], bins=30, label=f'Erro IA Fase {fase}', color=cores['ia'][i], kde=True, ax=axs[1, i])
        axs[1, i].set_title(f'Distribuição dos Erros em relação ao EE - Fase {fase}')
        axs[1, i].set_xlabel('Erro Absoluto (p.u.)')
        axs[1, i].set_ylabel('Frequência')
        axs[1, i].legend()

    for ax in axs[0, :]:
        ax.tick_params(labelbottom=True)

    plt.tight_layout()
    plt.show()

# Função para criar scatter plots com correlações
def plot_scatter(ax, x_data, y_data, titulo, xlabel, ylabel, cor, texto_correlacao):
    ax.scatter(x_data, y_data, alpha=0.5, color=cor)
    ax.plot([x_data.min(), x_data.max()], [x_data.min(), x_data.max()], 'r--')
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.05, 0.9, texto_correlacao, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            edgecolor='black', facecolor='white'))

# Obter o caminho do diretório atual
CurrentFolder = Path().resolve()
MasterFile = CurrentFolder / 'objs' / '13Bus' / 'Previsoes' / 'resultados_IA_V2.csv'
data = pd.read_csv(MasterFile)

# Filtrar os dados para a barra 634
barra_634 = data[data['nome_barra'] == '634']

hora_inicial = int(re.findall(r'\d+', data['hora'][0])[0])
hora_final = int(re.findall(r'\d+', data['hora'][data.shape[0]-1])[0])

total_horas = hora_final - hora_inicial

# Cálculo das métricas para cada fase
metricas = {}
for fase in range(3):
    metricas[fase] = {
        'Medidor ~ EE': calcular_metricas(barra_634[f'tensao_{fase}'], barra_634[f'tensao_estimada_{fase}']),
        'Medidor ~ IA': calcular_metricas(barra_634[f'tensao_{fase}'], barra_634[f'tensao_estimada_{fase}_pred']),
        'Medidor ~ Gabarito': calcular_metricas(barra_634['tensao_{fase}'], barra_634['tensao_gabarito_{fase}']),
        'EE ~ IA': calcular_metricas(barra_634[f'tensao_estimada_{fase}'], barra_634[f'tensao_estimada_{fase}_pred'])
    }

# Plotar o gráfico
fig, ax = plt.subplots(figsize=(12, 8))
plotar_tensoes(barra_634, 0, ax)

metrics_estimador_text = {
    'RMSE (Medidor ~ EE)': metricas[0]['Medidor ~ EE'][0],
    'Erro Médio Absoluto (Medidor ~ EE)': metricas[0]['Medidor ~ EE'][1],
    'Correlação (Medidor ~ EE)': metricas[0]['Medidor ~ EE'][2]
}

metrics_ia_text = {
    'RMSE (Medidor ~ IA)': metricas[0]['Medidor ~ IA'][0],
    'Erro Médio Absoluto (Medidor ~ IA)': metricas[0]['Medidor ~ IA'][1],
    'Correlação (Medidor ~ IA)': metricas[0]['Medidor ~ IA'][2]
}

metrics_estimador_ia_text = {
    'RMSE (EE ~ IA)': metricas[0]['EE ~ IA'][0],
    'Erro Médio Absoluto (EE ~ IA)': metricas[0]['EE ~ IA'][1],
    'Correlação (EE ~ IA)': metricas[0]['EE ~ IA'][2]
}

exibir_metricas(ax, 'Métricas Medidor ~ EE Fase 0', metrics_estimador_text, [0.02, 0.98])
exibir_metricas(ax, 'Métricas Medidor ~ IA Fase 0', metrics_ia_text, [0.02, 0.88])
exibir_metricas(ax, 'Métricas EE ~ IA Fase 0', metrics_estimador_ia_text, [0.02, 0.78])

plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=72))  # Mostrar uma marcação a cada 7 dias
plt.title(f'Comparação dos valores de tensão na Fase 0 da Barra 634 ao longo de {total_horas} horas')
plt.xlabel('Horas')
plt.xticks(rotation=90)
plt.ylabel('Tensão (p.u.)')
plt.legend(loc='upper right')

plt.tight_layout()  # Ajusta automaticamente o layout
plt.show()

# Uso das funções
fases = [0, 1, 2]
erros = calcular_erros(barra_634, fases)
plotar_distribuicao_erros(erros, fases)

# Definir cores diferentes para cada fase
cores = ['blue', 'green', 'orange']

# Criar a figura e os subplots de 3 linhas e 3 colunas
fig, axes = plt.subplots(3, 3, figsize=(10, 8))

# Loop para cada fase (0, 1, 2)
for i, fase in enumerate(fases):
    tensao_col = f'tensao_{fase}'
    tensao_estimada_col = f'tensao_estimada_{fase}'
    tensao_ia_col = f'tensao_estimada_{fase}_pred'

    # Correlação IA vs Medidor
    corr_ia_text = f'Correlação na fase {fase}: {metricas[fase]["Medidor ~ IA"][2]:.6f}'
    plot_scatter(axes[i, 0], barra_634[tensao_col], barra_634[tensao_ia_col], 
                 f'Correlação IA vs Medidor (Fase {fase} - Barra 634)', 
                 'Tensão Medidor (p.u.)', 'Tensão IA (p.u.)', cores[i], corr_ia_text)
    
    # Correlação IA vs Estimador
    corr_estimada_ia_text = f'Correlação na fase {fase}: {metricas[fase]["EE ~ IA"][2]:.6f}'
    plot_scatter(axes[i, 1], barra_634[tensao_estimada_col], barra_634[tensao_ia_col], 
                 f'Correlação IA vs Estimador (Fase {fase} - Barra 634)', 
                 'Tensão Estimador (p.u.)', 'Tensão IA (p.u.)', cores[i], corr_estimada_ia_text)

    # Correlação Estimador vs Medidor
    corr_estimada_text = f'Correlação na fase {fase}: {metricas[fase]["Medidor ~ EE"][2]:.6f}'
    plot_scatter(axes[i, 2], barra_634[tensao_col], barra_634[tensao_estimada_col], 
                 f'Correlação Estimador vs Medidor (Fase {fase} - Barra 634)', 
                 'Tensão Medidor (p.u.)', 'Tensão Estimada (p.u.)', cores[i], corr_estimada_text)

# Ajustar o layout
plt.tight_layout()
plt.show()