import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.dates as mdates

# Obter o caminho do diretório atual
CurrentFolder = Path().resolve()
MasterFile = CurrentFolder / 'objs' / '13Bus' / 'Previsoes' / 'resultados_IA.csv'
data = pd.read_csv(MasterFile)

# Filtrar os dados para a barra 634
barra_634 = data[data['nome_barra'] == '634']

# Calcular as métricas entre medidor e tensão estimada
mse_estimada = mean_squared_error(barra_634['tensao_0'], barra_634['tensao_estimada_0'])
rmse_estimada = np.sqrt(mse_estimada)
mae_estimada = mean_absolute_error(barra_634['tensao_0'], barra_634['tensao_estimada_0'])
corr_estimada, _ = pearsonr(barra_634['tensao_0'], barra_634['tensao_estimada_0'])

# Calcular as métricas entre medidor e tensao prevista pela IA
mse_ia = mean_squared_error(barra_634['tensao_0'], barra_634['tensao_estimada_0_pred'])
rmse_ia = np.sqrt(mse_estimada)
mae_ia = mean_absolute_error(barra_634['tensao_0'], barra_634['tensao_estimada_0_pred'])
corr_ia, _ = pearsonr(barra_634['tensao_0'], barra_634['tensao_estimada_0_pred'])

# Calcular as métricas entre medidor e tensao gabarito
mse_gabarito = mean_squared_error(barra_634['tensao_0'], barra_634['tensao_gabarito_0'])
rmse_gabarito = np.sqrt(mse_estimada)
mae_gabarito = mean_absolute_error(barra_634['tensao_0'], barra_634['tensao_gabarito_0'])
corr_gabarito, _ = pearsonr(barra_634['tensao_0'], barra_634['tensao_gabarito_0'])

# Calcular as métricas entre tensão estimada e tensao prevista pela IA
mse_estimada_ia = mean_squared_error(barra_634['tensao_estimada_0'], barra_634['tensao_estimada_0_pred'])
rmse_estimada_ia = np.sqrt(mse_estimada)
mae_estimada_ia = mean_absolute_error(barra_634['tensao_estimada_0'], barra_634['tensao_estimada_0_pred'])
corr_estimada_ia, _ = pearsonr(barra_634['tensao_estimada_0'], barra_634['tensao_estimada_0_pred'])

# Plotar o gráfico
plt.figure(figsize=(12, 8))
plt.plot(barra_634['hora'], barra_634['tensao_0'], label='Tensão 0')
plt.plot(barra_634['hora'], barra_634['tensao_estimada_0'], label='Tensão Estimada 0', linestyle='--')
plt.plot(barra_634['hora'], barra_634['tensao_estimada_0_pred'], label='Tensão Prevista IA 0', linestyle='--')
plt.plot(barra_634['hora'], barra_634['tensao_gabarito_0'], label='Tensão Gabarito 0', linestyle='--')

# Configurar o eixo x
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=72))  # Mostrar uma marcação a cada 7 dias
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Formato da data

# Adicionar as métricas ao gráfico
metrics_estimada_text = f'RMSE (Medidor ~ EE): {rmse_estimada:.6f}\nErro Médio Absoluto (Medidor ~ EE): {mae_estimada:.6f}\nCorrelação (Medidor ~ EE): {corr_estimada:.6f}'
metrics_ia_text = f'RMSE (Medidor ~ IA): {rmse_ia:.6f}\nErro Médio Absoluto (Medidor ~ IA): {mae_ia:.6f}\nCorrelação (Medidor ~ IA): {corr_ia:.6f}'
metrics_gabarito_text = f'RMSE (Medidor ~ Gabarito): {rmse_gabarito:.6f}\nErro Médio Absoluto (Medidor ~ Gabarito): {mae_gabarito:.6f}\nCorrelação (Medidor ~ Gabarito): {corr_gabarito:.6f}'
metrics_estimada_ia_text = f'RMSE (EE ~ IA): {rmse_estimada_ia:.6f}\nErro Médio Absoluto (EE ~ IA): {mae_estimada_ia:.6f}\nCorrelação (EE ~ IA): {corr_estimada_ia:.6f}'

plt.text(0.02, 0.98, metrics_estimada_text, transform=plt.gca().transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.text(0.02, 0.88, metrics_ia_text, transform=plt.gca().transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.text(0.02, 0.78, metrics_gabarito_text, transform=plt.gca().transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.text(0.02, 0.68, metrics_estimada_ia_text, transform=plt.gca().transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.title('Comparação dos valores de tensão na Fase 0 da Barra 634 ao longo de 4830 horas')
plt.xlabel('Horas')
plt.xticks(rotation=90)
plt.ylabel('Tensão (p.u.)')
plt.legend(loc='upper right')

plt.tight_layout()  # Ajusta automaticamente o layout
plt.show()