from pathlib import Path
import numpy as np
import pandas as pd
import time
import EESD

nivel_ruido = 'normal'

""" Especificação do ruído na medida """

if nivel_ruido == "normal":
    
    meas_noise = 0.02 # Coeficiente de ruído na medida de potência
    v_noise = 0.01 # Coeficiente de ruído na medida de tensão
    i_noise = 0.01
    zero_inj_coef = 0.001
    
    load_error = 0.2 # Coeficiente de erro na carga
    sgen_error = 0.15 # Coeficiente de erro na GD

    pm_noise = load_error/2 
    sgen_noise = sgen_error/2

if nivel_ruido == "ruim":
    
    meas_noise = 0.05 # Coeficiente de ruído na medida de potência
    v_noise = 0.03 # Coeficiente de ruído na medida de tensão
    i_noise = 0.03
    zero_inj_coef = 0.001
    
    load_error = 0.3 # Coeficiente de erro na carga
    sgen_error = 0.2 # Coeficiente de erro na GD

    pm_noise = load_error/2  
    sgen_noise = sgen_error/2
 
if nivel_ruido == "bom":
    
    meas_noise = 0.01 # Coeficiente de ruído na medida de potência
    v_noise = 0.005 # Coeficiente de ruído na medida de tensão
    i_noise = 0.005
    zero_inj_coef = 0.001

    load_error = 0.15 # Coeficiente de erro na carga
    sgen_error = 0.1 # Coeficiente de erro na GD
    
    pm_noise = load_error/2  
    sgen_noise = sgen_error/2


""" Perfis de carga de 24h """
if True: # Para reduzir o tamanho aparente do código
    perfil_day_pq = np.array([0.2,0.2,0.2,0.2,0.3,0.3,
                            0.4,0.4,0.5,0.5,0.7,0.8,
                            0.9,1.0,1.0,1.0,0.9,0.8,
                            0.8,0.7,0.7,0.6,0.5,0.3])

    perfil_day_sun = np.array([0.0,0.00,0.0,0.0,0.0,0.0,
                                0.1,0.25,0.4,0.7,0.9,1.0,
                                1.0,1.00,1.0,1.0,0.9,0.8,
                                0.6,0.40,0.3,0.1,0.0,0.0])

    perfil_day_wind = np.array([0.6,0.6,0.7,0.5,0.4,0.4,
                                0.5,0.7,0.8,0.7,0.5,0.5,
                                0.4,0.5,0.4,0.5,0.6,0.6,
                                0.3,0.4,0.7,0.6,0.4,0.5])

    industry_load_perfil = np.array([0.35,0.35,0.30,0.30,0.40,0.50,
                                    0.60,0.90,1.00,1.00,1.00,0.90,
                                    0.85,0.85,0.85,0.85,0.80,0.55,
                                    0.50,0.45,0.40,0.40,0.35,0.35])

    household_load_perfil = np.array([0.25,0.20,0.20,0.20,0.20,0.25,
                                    0.40,0.65,0.65,0.65,0.70,0.60,
                                    0.70,0.65,0.55,0.50,0.45,0.60,
                                    0.80,0.90,0.80,0.70,0.55,0.30])
    
def get_gabarito(eesd: EESD.EESD) -> np.array:
    ang = np.array([])
    tensoes = np.array([])
    for barra in eesd.DSSCircuit.AllBusNames:
        eesd.DSSCircuit.SetActiveBus(barra)
        ang = np.concatenate([ang, eesd.DSSCircuit.Buses.puVmagAngle[1::2]*2*np.pi / 360])
        tensoes = np.concatenate([tensoes, eesd.DSSCircuit.Buses.puVmagAngle[::2]])

    return np.concatenate([ang[3:], tensoes[3:]])

def get_gabarito_anual(eesd: EESD.EESD, total_horas: int) -> dict:
    gabarito_dict = {}
    for h in range(total_horas):
        ang = np.array([])
        tensoes = np.array([])
        for barra in eesd.DSSCircuit.AllBusNames:
            eesd.DSSCircuit.Solution.Solve()
            eesd.DSSCircuit.SetActiveBus(barra)
            ang = np.concatenate([ang, eesd.DSSCircuit.Buses.puVmagAngle[1::2]*2*np.pi / 360])
            tensoes = np.concatenate([tensoes, eesd.DSSCircuit.Buses.puVmagAngle[::2]])
        gabarito_dict[f"hora_{h}"] = np.concatenate([ang[3:], tensoes[3:]])
    return gabarito_dict

def analise_gabarito(gabarito, vet_estados, completa: bool = False):
    tam = len(gabarito['hora_0'])
    maior_erro_tensao = {}
    maior_erro_angulo = {}
    for hora in gabarito.keys():
        ang_gab = gabarito[hora][:tam//2]
        ten_gab = gabarito[hora][tam//2:]
        ang_vet = vet_estados[hora][:tam//2]
        ten_vet = vet_estados[hora][tam//2:]
        erro_ang = max(abs(ang_vet-ang_gab))*360/(2*np.pi)
        erro_tensao = max(abs(ten_vet-ten_gab)/ten_gab)
        maior_erro_angulo[f'{hora}'] = erro_ang
        maior_erro_tensao[f'{hora}'] = erro_tensao
        if completa:
            print(f'O maior erro absoluto de ângulos na {hora} foi de {round(erro_ang,3)} graus')
            print(f'O maior erro relativo de tensão na {hora} foi de {round(erro_tensao*100,3)}%')
    
    hora_ang = max(maior_erro_angulo.keys(), key=(lambda key: maior_erro_angulo[key]))
    hora_tensao = max(maior_erro_tensao.keys(), key=(lambda key: maior_erro_tensao[key]))
    print(f'O maior erro absoluto de ângulos na {hora_ang} foi de {round(maior_erro_angulo[hora_ang],3)} graus')
    print(f'O maior erro relativo de tensão na {hora_tensao} foi de {round(maior_erro_tensao[hora_tensao]*100,3)}%')

def distribuir_valores(df, vet_estados):
    # Valores fixos para a barra sourcebus
    angulos_sourcebus = [0, 2 * np.pi / 3, -2 * np.pi / 3]
    tensoes_sourcebus = [1, 1, 1]
    
    # Determinar a quantidade de ângulos e tensões
    num_angulos = sum(len(fases) for nome, fases in zip(df['nome_barra'], df['Fases']) if nome != 'sourcebus')
    num_tensoes = num_angulos

    # Separar a lista de estados
    angulos = vet_estados[:num_angulos]
    tensoes = vet_estados[num_angulos:num_angulos + num_tensoes]

    # Função interna para distribuir valores
    def distribuir(df, valores, col_name, fixed_values=None):
        idx = 0
        result = []
        for nome_barra, fases in zip(df['nome_barra'], df['Fases']):
            if nome_barra == 'sourcebus':
                result.append(fixed_values)
            else:
                n = len(fases)
                result.append(valores[idx:idx + n])
                idx += n
        df[col_name] = result

    # Distribuir ângulos e tensões
    distribuir(df, angulos, 'Ângulo de Tensão Estimado', fixed_values=angulos_sourcebus)
    distribuir(df, tensoes, 'Tensão Estimada', fixed_values=tensoes_sourcebus)

    return df

def distribuir_valores_horarios(df:pd.DataFrame, vet_estados_anual:dict, gabarito_anual:dict) -> pd.DataFrame:
    # Inicializa as listas para armazenar os dicionários de ângulos e tensões
    angulos_sourcebus = [0, 2*np.pi/3, -2*np.pi/3]
    tensoes_sourcebus = [1, 1, 1]
    angulos_estimados_col = []
    tensoes_estimadas_col = []
    angulos_gabarito_col = []
    tensoes_gabarito_col = []
    idx = 0
    # Itera sobre as linhas do DataFrame
    for _, row in df.iterrows():
        nome_barra = row['nome_barra']
        fases = row['Fases']

        # Inicializa dicionários para armazenar valores por hora
        angulos_estimados_dict = {}
        tensoes_estimadas_dict = {}
        angulos_gabarito_dict = {}
        tensoes_gabarito_dict = {}
        num_fases = len(fases)
        num_barras = len(vet_estados_anual["hora_0"])//2
        
        def itera_sobre_vetor(dicionario:dict, angulos_estimados_dict, tensoes_estimadas_dict, idx, num_fases, num_barras):
            # Itera sobre as horas no vetor de estados anual
            for hora, valores_estimados in dicionario.items():

                if nome_barra == 'sourcebus':
                    angulos_estimados_dict[hora] = angulos_sourcebus
                    tensoes_estimadas_dict[hora] = tensoes_sourcebus
                else:
                    # Extrai os valores correspondentes
                    angulos_estimados_dict[hora] = np.array(valores_estimados[idx:num_fases+idx])
                    tensoes_estimadas_dict[hora] = np.array(valores_estimados[num_barras+idx:num_barras*2])

            return angulos_estimados_dict, tensoes_estimadas_dict
        
        angulos_estimados_dict, tensoes_estimadas_dict = itera_sobre_vetor(vet_estados_anual, angulos_estimados_dict, tensoes_estimadas_dict, idx, num_fases, num_barras)
        angulos_gabarito_dict, tensoes_gabarito_dict = itera_sobre_vetor(gabarito_anual, angulos_gabarito_dict, tensoes_gabarito_dict, idx, num_fases, num_barras)

        idx += num_fases
        # Adiciona os dicionários às colunas correspondentes
        angulos_estimados_col.append(angulos_estimados_dict)
        tensoes_estimadas_col.append(tensoes_estimadas_dict)
        angulos_gabarito_col.append(angulos_gabarito_dict)
        tensoes_gabarito_col.append(tensoes_gabarito_dict)

    df['angulo_estimado'] = angulos_estimados_col
    df['tensao_estimada'] = tensoes_estimadas_col
    df['angulo_gabarito'] = angulos_gabarito_col
    df['tensao_gabarito'] = tensoes_gabarito_col

    return df

def formatar_dataframe(df):
    linhas_formatadas = []

    for hora in df.iloc[0]['angulo_estimado'].keys():
        for _, row in df.iterrows():
            nome_barra = row['nome_barra']
            fases = row['Fases']
            bases = row["Bases"]
            geracao = row["Geracao"]
            inj_pot_at = row["Inj_pot_at"][hora]
            inj_pot_rat = row["Inj_pot_rat"][hora]
            tensao = row["Tensao"][hora]
            angulos_estimados = row['angulo_estimado'][hora]
            tensoes_estimadas = row['tensao_estimada'][hora]
            angulos_gabarito = row['angulo_gabarito'][hora]
            tensoes_gabarito = row['tensao_gabarito'][hora]

            inj_pot_at_dict = {f"inj_pot_at_{i}": inj_pot_at[i] if inj_pot_at[i] != 0 else None for i in range(len(inj_pot_at))}
            inj_pot_rat_dict = {f"inj_pot_rat_{i}": inj_pot_rat[i] if inj_pot_rat[i] != 0 else None for i in range(len(inj_pot_rat))}
            tensao_dict = {f"tensao_{i}": tensao[i] if tensao[i] != 0 else None for i in range(len(tensao))}
            fase_dict = {f"Fase_{i}": 1 if i in fases else 0 for i in range(3)}

            ang_tensao_estimado_dict = {}
            tensao_estimada_dict = {}

            ang_tensao_gabarito_dict = {}
            tensao_gabarito_dict = {}

            for i, fase in enumerate(fases):
                ang_tensao_estimado_dict[f"ang_tensao_estimado_{fase}"] = angulos_estimados[i] if i < len(angulos_estimados) else None
                tensao_estimada_dict[f"tensao_estimada_{fase}"] = tensoes_estimadas[i] if i < len(tensoes_estimadas) else None
                ang_tensao_gabarito_dict[f"ang_tensao_gabarito_{fase}"] = angulos_gabarito[i] if i < len(angulos_gabarito) else None
                tensao_gabarito_dict[f"tensao_gabarito_{fase}"] = tensoes_gabarito[i] if i < len(tensoes_gabarito) else None

            nova_linha = {
                'hora': hora,
                'nome_barra': nome_barra,
                'base': bases,
                'Geracao': 1 if geracao else None,
                **fase_dict,
                **inj_pot_at_dict,
                **inj_pot_rat_dict,
                **tensao_dict,
                **ang_tensao_estimado_dict,
                **tensao_estimada_dict,
                **ang_tensao_gabarito_dict,
                **tensao_gabarito_dict
            }

            linhas_formatadas.append(nova_linha)

    df_formatado = pd.DataFrame(linhas_formatadas)
    return df_formatado

def salvar_dataframe_como_csv(df_formatado):
    # Definir o caminho para salvar o arquivo CSV
    path = Path(__file__)
    CurrentFolder = path.parent
    MasterFile = CurrentFolder / 'objs' / '13Bus' / 'Datasets' / 'dataset_anual.csv'

    # Criar os diretórios, se necessário
    MasterFile.parent.mkdir(parents=True, exist_ok=True)

    # Salvar o dataframe como CSV
    df_formatado.to_csv(MasterFile, index=False)

def main():
    #Achar o path do script do OpenDSS
    path = Path(__file__)
    CurrentFolder = path.parent
    MasterFile = CurrentFolder / 'objs' / '13Bus' / 'IEEE13Nodeckt_com_loadshape_redirecionada.dss'
    
    'IEEE13Nodeckt_com_loadshape.dss'
    '4_SEAUA_1''Master_DU01_20201246_4_SEAUA_1_NTMBSR1PVTTR.dss'
    'Sulgipe''Master_DU01_20201246_1_SEAUA_1_NTMBSR1PVTTR.dss'
    
    verbose = False

    medidas_imperfeitas = True
    
    baseva =  33.3 * 10**6

    total_horas = 24

    eesd = EESD.EESD(MasterFile, total_horas, baseva, verbose, medidas_imperfeitas)
    
    inicio = time.time()
    vet_estados = eesd.run(10**(-5), 100)
    vet_estados_anual = eesd.run_anual(10**(-5), 100, total_horas)

    fim = time.time()
    
    print(f'Estimador concluido em {fim-inicio}s')
    
    gabarito = get_gabarito(eesd)
    gabarito_anual = get_gabarito_anual(eesd, total_horas)

    if verbose:
        print(f"Gabarito: {gabarito}")

    return EESD.EESD.medidas(eesd,baseva), EESD.EESD.medidas_anuais(eesd,baseva, total_horas), vet_estados, vet_estados_anual, total_horas, gabarito_anual
    
if __name__ == '__main__':
    medidas, medidas_anuais, vet_estados, vet_estados_anual, total_horas, gabarito_anual = main()

data_medidas = medidas[0]
data_medidas_anuais = medidas_anuais[0]

dataframe_completo = distribuir_valores(data_medidas, vet_estados)

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #print(dataframe_completo)

dataframe_completo_anual = distribuir_valores_horarios(data_medidas_anuais, vet_estados_anual, gabarito_anual)

#print(dataframe_completo_anual)

df_formatado = formatar_dataframe(dataframe_completo_anual)

salvar_dataframe_como_csv(df_formatado)


