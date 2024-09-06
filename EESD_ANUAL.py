import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse as scsp

import time

from dss import DSS as dss_engine
from Jacobiana import Jacobiana
from Residuos import Residuo

import networkx as nx
import matplotlib.pyplot as plt

import threading

class EESD_ANUAL():
    def __init__(self, master_path, total_horas, hora_inicial, hora_final, baseva: float = 10**6, verbose: bool = False, medidas_imperfeitas: bool = False) -> None:
        self.DSSCircuit, self.DSSText, self.DSSObj, self.DSSMonitors = self.InitializeDSS()
        self.baseva = baseva
        self.MasterFile = master_path
        self.verbose = verbose
        self.hora_inicial = hora_inicial
        self.hora_final = hora_final
        self.medidas_imperfeitas = medidas_imperfeitas

        self.total_horas = total_horas

        self.resolve_fluxo_carga()
        print('Acabou o fluxo de carga')

        self.barras, self.num_medidas = self.medidas(self.baseva)
        self.barras_anuais, self.num_medidas_anuais = self.medidas_anuais(self.baseva, self.total_horas)

        self.vet_estados_anuais = self.iniciar_vet_estados_anuais(self.hora_inicial, self.hora_final)
        
        print('Vetor de estados iniciado')

        Ybus = scsp.csr_matrix(self.DSSObj.YMatrix.GetCompressedYMatrix())
        Ybus = scsp.lil_matrix(Ybus)
        
        self.Ybus, self.nodes = self.organiza_Ybus(Ybus)

        self.Ybus = self.Conserta_Ybus(self.Ybus)
        self.Gs = np.real(self.Ybus).toarray()
        self.Bs = np.imag(self.Ybus).toarray()
        print('Matriz de adimitância modificada com sucesso')

        self.count = self.barras['Geracao'].value_counts().iloc[1]
        fases = self.barras['Fases'].tolist()
        self.fases = [sub_elem for elem in fases for sub_elem in elem]
        

    def plotar_sistema(self) -> plt:
        dss_engine.Plotting.enable(plot3d=True, plot2d=True)
        dss_engine.Text.Command = 'plot circuit labels=y'

        # Coordenadas de cada barra (bus)
        barras = {
            '650': (2, 4), '632': (2, 3), '633': (3, 3), '634': (4, 3),
            '645': (1, 3), '646': (0, 3), '671': (2, 2), '684': (1, 2),
            '692': (3, 2), '675': (4, 2), '680': (2, 1), '652': (1, 1), '611': (0, 2)
        }

        # Linhas entre as barras
        linhas = [
            ('650', '632'), ('632', '633'), ('633', '634'), ('632', '645'), ('645', '646'),
            ('632', '671'), ('671', '684'), ('684', '611'), ('671', '680'), ('680', '652'),
            ('671', '692'), ('692', '675')
        ]

        # Cria o diagrama
        fig, ax = plt.subplots()

        # Desenhar linhas entre barras
        for linha in linhas:
            bus1, bus2 = linha
            x_values = [barras[bus1][0], barras[bus2][0]]
            y_values = [barras[bus1][1], barras[bus2][1]]
            ax.plot(x_values, y_values, 'k-', lw=2)

        # Desenhar as barras como nós
        for bus, coord in barras.items():
            ax.plot(coord[0], coord[1], 'ko', markersize=10)
            # Ajustar o posicionamento do texto
            if bus in ['650', '646', '645', '633', '634', '611', '684', '692', '675', '652']:
                ax.text(coord[0], coord[1] + 0.2, f'{bus}', fontsize=12, ha='center', va='top')  # Acima do nó
            elif bus in ['632', '671']:
                ax.text(coord[0] + 0.15, coord[1] + 0.15, f'{bus}', fontsize=12, ha='center', va='center')  # À direita do nó
            elif bus in ['680']:
                ax.text(coord[0], coord[1] - 0.2, f'{bus}', fontsize=12, ha='center', va='bottom')  # À esquerda do nó


        # Configurar limites e título
        ax.set_xlim(-1, 5)
        ax.set_ylim(0, 5)
        ax.set_aspect('equal')
        ax.set_title('Diagrama Unifilar da Rede')

        plt.show()

    def resolve_fluxo_carga(self):
        self.DSSText.Command = 'Clear'
        self.DSSText.Command = f'Compile {self.MasterFile}'

        self.iniciar_medidores()

        self.DSSText.Command = 'Solve'

    def InitializeDSS(self) -> tuple:
        DSSObj = dss_engine
        flag = DSSObj.Start(0)
        if flag:
            print('OpenDSS COM Interface initialized succesfully.')
            
        else:
            print('OpenDSS COMInterface failed to start.')
            
        #Set up the interface variables - Comunication OpenDSS with Python
        DSSText = DSSObj.Text
        DSSCircuit = DSSObj.ActiveCircuit
        DSSMonitors = DSSCircuit.Monitors

        return DSSCircuit, DSSText, DSSObj, DSSMonitors

    def iniciar_medidores(self) -> None:
        count = 0
        for i, barra in enumerate(self.DSSCircuit.AllBusNames):
            self.DSSCircuit.SetActiveBus(barra)
            for j, elem in enumerate(self.DSSCircuit.Buses.AllPCEatBus):
                if 'Load' in elem or 'Generator' in elem or 'Vsource' in elem or 'PVSystem' in elem:
                    self.DSSText.Command = f'New Monitor.pqi{count} element={elem}, terminal=1, mode=1, ppolar=no'
                    count += 1
                    
            max_fases = 0
            elem = 'None'
            for pde in self.DSSCircuit.Buses.AllPDEatBus:
                self.DSSCircuit.SetActiveElement(pde)
                num_fases = len(self.DSSCircuit.ActiveCktElement.NodeOrder)
                if num_fases > max_fases:
                    elem = self.DSSCircuit.ActiveCktElement.Name
                    max_fases = num_fases
            if elem != 'None':
                self.DSSCircuit.SetActiveElement(elem)
                if self.DSSCircuit.ActiveCktElement.BusNames[0].split('.')[0] == barra:
                    self.DSSText.Command = f'New Monitor.v{i} element={elem}, terminal=1, mode=32'
                    
                elif self.DSSCircuit.ActiveCktElement.BusNames[1].split('.')[0] == barra:
                    self.DSSText.Command = f'New Monitor.v{i} element={elem}, terminal=2, mode=32'
                    
                else:
                    print(f'Nenhum elemento conectado na barra {barra}')

    def indexar_barras(self) -> pd.DataFrame:
        #Designa indíces às barras
        nomes = []
        bases = []
        geracao = []
        for barra in self.DSSCircuit.AllBusNames:
            #if barra.isdigit(): é possível que o sourcebus e o reg não entrem para a EE
            self.DSSCircuit.SetActiveBus(barra)
            #Base é em fase-neutro
            base = self.DSSCircuit.Buses.kVBase
            if base == 0:
                raise ValueError('Tensão base não pode ser 0')
            nomes.append(barra)
            bases.append(base)
            geracao.append(self.DSSCircuit.Buses.AllPCEatBus[0] == 'Vsource.source')

        nomes = np.concatenate([nomes[1:], [nomes[0]]])
        bases = np.concatenate([bases[1:], [bases[0]]])
        geracao = np.concatenate([geracao[1:], [geracao[0]]])

        idx = [i for i in range(len(nomes))]
        
        barras = pd.DataFrame(columns=['nome_barra', 'Bases', 'Fases', 'Inj_pot_at', 'Inj_pot_rat', 'Tensao', 'Inj_pot_at_est', 'Inj_pot_rat_est', 'Geracao'],
                            index=idx)
        
        barras['nome_barra'] = nomes
        barras.loc[idx, 'Bases'] = bases
        barras.loc[idx, 'Geracao'] = geracao

        return barras

    def gera_medida_imperfeita(self) -> None:
        # Gerar fatores aleatórios com base na distribuição normal
        fatores_anual = {}
        for hora in range(self.total_horas):
            fatores = np.random.normal(0, self.dp_anual[f'hora_{hora}'], len(self.medidas_anual[f'hora_{hora}']))
            fatores_anual[f'hora_{hora}'] = fatores
            self.medidas_anual[f'hora_{hora}'] = self.medidas_anual[f'hora_{hora}'] + fatores_anual[f'hora_{hora}']
            medidas = self.medidas_anual[f'hora_{hora}']
    
    def iniciar_vet_estados_anuais(self, hora_inicial:int, hora_final:int) -> dict:
        fases = self.barras['Fases'].tolist()
        fases = [sub_elem for elem in fases for sub_elem in elem]
        tensoes = np.array([1 for _ in fases[:-3]], dtype=np.float64)
        angulos = np.zeros(len(fases[:-3]))
        
        for i, fase in enumerate(fases[:-3]):
            if fase == 1:
                angulos[i] = -120 * 2 * np.pi / 360
            elif fase == 2:
                angulos[i] = 120 * 2 * np.pi / 360
        
        vet_estados = np.concatenate((angulos, tensoes))
        vet_estados_anuais = {f"hora_{h}": vet_estados for h in range(hora_inicial, hora_final)}
        return vet_estados_anuais
    
    def achar_index_barra(self, barras: pd.DataFrame, barra: int) -> int:
        #Retorna o index da barra do monitor ativo
        self.DSSCircuit.SetActiveElement(self.DSSMonitors.Element)
        
        self.DSSCircuit.SetActiveBus(self.DSSCircuit.ActiveCktElement.BusNames[barra])
        nome = self.DSSCircuit.Buses.Name
        
        return barras.index[barras['nome_barra'] == nome].to_list()[0]

    def pegar_fases(self) -> np.array:
        fases = self.DSSCircuit.ActiveBus.Nodes - 1
        fases = list(dict.fromkeys(fases))
        fases = [fase for fase in fases if fase != -1]
        
        return fases

    def medidas(self, baseva: int) -> pd.DataFrame:
        barras = self.indexar_barras()
        erro_aleatorio_medidor_pot = np.random.choice(np.arange(0.01, 0.021, 0.001))
        erro_aleatorio_medidor_tensao = np.random.choice(np.arange(0.002, 0.011, 0.001))

        dp_padrao_pot = erro_aleatorio_medidor_pot/(3*100) # Tese da Livia + Orientação do prof Alex
        dp_padrao_tensao = erro_aleatorio_medidor_tensao/(3*100) # Tese da Livia + Orientação do prof Alex

        num_medidas = 0
        for idx, bus in enumerate(barras['nome_barra']):
            self.DSSCircuit.SetActiveBus(bus)
            fases = self.pegar_fases()
            #Preenche a coluna das fases
            barras.iat[idx, 2] = fases
            
            if not barras['Geracao'][idx]:
                #Inicializa as medidas como 0 para injeção ativa e reativa
                barras.iat[idx, 3] = np.array([0, 0, 0, 0], dtype=np.float64)
                barras.iat[idx, 4] = np.array([0, 0, 0, 0], dtype=np.float64)
                num_medidas += len(fases)*2
        
        #Amostra e salva os valores dos medidores do sistema
        self.DSSMonitors.SampleAll()
        self.DSSMonitors.SaveAll()

        self.DSSMonitors.First
        for _ in range(self.DSSMonitors.Count):
            barra = self.DSSMonitors.Terminal - 1
            index_barra = self.achar_index_barra(barras, barra)
            
            #Pegar as fases da carga atual
            fases = self.DSSCircuit.ActiveCktElement.NodeOrder - 1
            fases = list(dict.fromkeys(fases))
            fases = [fase for fase in fases if fase != -1]
            matriz_medidas = self.DSSMonitors.AsMatrix()[0][2:]
            
            if 'pqi' in self.DSSMonitors.Name:
                fatores_pot = [1,1,1,1]
                if self.medidas_imperfeitas:
                    fatores_pot = np.random.normal(1, dp_padrao_pot, 4)
                    fatores_pot = np.clip(fatores_pot, 0.995, 1.005)
                medidas_at = np.zeros(4)
                medidas_rat = np.zeros(4)
                
                for i, fase in enumerate(fases):
                    medidas_at[fase] = matriz_medidas[i*2]
                    medidas_rat[fase] = matriz_medidas[i*2+1]

                # Multiplicar ponto a ponto por random de media 1 e 4 itens
                #barras.loc[[index_barra], 'Inj_pot_at'] += pd.Series([(-medidas_at*1000 / baseva)*fatores_pot], index=barras.index[[index_barra]])
                #barras.loc[[index_barra], 'Inj_pot_rat'] += pd.Series([(-medidas_rat*1000 / baseva)*fatores_pot], index=barras.index[[index_barra]])
                
                #Atualiza as medidas de injeção ativa e reativa
                barras.iat[index_barra, 3] += (-medidas_at*1000 / baseva)*fatores_pot
                barras.iat[index_barra, 4] += (-medidas_rat*1000 / baseva)*fatores_pot

            elif 'v' in self.DSSMonitors.Name:
                if type(barras['Tensao'][index_barra]) != np.ndarray:
                    medidas = np.zeros(4)
                    
                    for i, fase in enumerate(fases):
                        medidas[fase] = matriz_medidas[i]

                    #Atualiza as medidas de tensão
                    basekv = self.DSSCircuit.Buses.kVBase

                    fatores_tensao = [1,1,1,1]
                    if self.medidas_imperfeitas:
                        fatores_tensao = np.random.normal(1, dp_padrao_tensao, 4)
                        fatores_tensao = np.clip(fatores_tensao, 0.995, 1.005)

                    barras.iat[index_barra, 5] = (medidas / (basekv*1000))*fatores_tensao
                    #barras.loc[[index_barra], 'Tensao'] = pd.Series([(medidas / (basekv*1000))*fatores_tensao], index=barras.index[[index_barra]])
                    if not barras['Geracao'][index_barra]:
                        num_medidas += len(fases)

            self.DSSMonitors.Next
        
        return barras, num_medidas

    def medidas_anuais(self, baseva:int, total_horas:int) -> pd.DataFrame:
        # Configurações do OpenDSS para simulação anual
        self.DSSText.Command = "set stepsize=1h"
        self.DSSText.Command = "set mode=yearly"
        self.DSSText.Command = "set number=1"

        instante_inicial, num_medidas = self.medidas(baseva)

        # Lista das colunas a serem zeradas
        colunas_para_zerar = ['Inj_pot_at', 'Tensao', 'Inj_pot_rat']

        # Aplicar a função para zerar os valores das colunas especificadas
        for coluna in colunas_para_zerar:
            instante_inicial[coluna] = instante_inicial[coluna].apply(lambda x: {})

        num_medidas_anual = num_medidas*total_horas
        for h in range(total_horas):
            self.DSSCircuit.Solution.Solve()

            # Captura as medições para o instante atual
            instante_atual = self.medidas(baseva)[0]

            for i, values in instante_atual.iterrows():
                # Adiciona ou atualiza o dicionário na coluna 'Inj_pot_at' do instante_inicial
                instante_inicial.at[i, 'Inj_pot_at'][f'hora_{h}'] = values['Inj_pot_at']
                instante_inicial.at[i, 'Inj_pot_rat'][f'hora_{h}'] = values['Inj_pot_rat']
                instante_inicial.at[i, 'Tensao'][f'hora_{h}'] = values['Tensao']

            self.DSSMonitors.ResetAll()

        return instante_inicial, num_medidas_anual 

    def forma_matriz(self, fases: np.ndarray, fases_barra: list, Yprim: list) -> np.array:
        temp = np.zeros((len(fases_barra), len(fases_barra)), dtype=np.complex128)
        fases = [fase for fase in fases if fase != -1]
        fases_barra = list(fases_barra-1)
        idx = []
        for fase in fases:
            idx.append(fases_barra.index(fase))
            
        k = 0
        for i in idx:
            for j in idx:
                temp[i, j] = Yprim[k]
                k += 1

        if len(Yprim) == 16:
            temp = np.reshape(Yprim, (4, 4))
        
        Yprim = temp
        
        return Yprim
    
    def organiza_Ybus(self, Ybus):

        nodes = {}
        for i, node in enumerate(self.DSSCircuit.YNodeOrder):
            nodes[node.lower()] = i

        temp = Ybus.copy()
        count = 0

        for i, bus in enumerate(self.DSSCircuit.AllBusNames):
            for fase in sorted(self.barras['Fases'].iloc[i-1]):
                no = nodes[f'{bus}.{fase+1}']
                temp[count] = Ybus[no].toarray()
                count += 1

        temp = temp.T
        Ybus_org = temp.copy()
        count = 0

        for i, bus in enumerate(self.DSSCircuit.AllBusNames):
            for fase in sorted(self.barras['Fases'].iloc[i-1]):
                no = nodes[f'{bus}.{fase+1}']
                Ybus_org[count] = temp[no]
                count += 1

        #csr_matrix pode ser mais rápida para sistemas maiores
        Ybus_org = scsp.lil_matrix(Ybus_org)
        nodes = {}
        count = 0

        for i, bus in enumerate(self.DSSCircuit.AllBusNames):
            for fase in sorted(self.barras['Fases'].iloc[i-1]):
                nodes[f'{bus}.{fase+1}'] = count

                count += 1
        return Ybus_org, nodes
    
    def Conserta_Ybus(self, Ybus):
        self.DSSCircuit.Transformers.First
        for _ in range(self.DSSCircuit.Transformers.Count):
            trafo = self.DSSCircuit.Transformers.Name
            self.DSSCircuit.SetActiveElement(trafo)
            num_phases = self.DSSCircuit.ActiveCktElement.NumPhases
            barras_conectadas = self.DSSCircuit.ActiveCktElement.BusNames
            self.DSSCircuit.SetActiveBus(barras_conectadas[0])
            basekv1 = self.DSSCircuit.Buses.kVBase
            self.DSSCircuit.SetActiveBus(barras_conectadas[1])
            basekv2 = self.DSSCircuit.Buses.kVBase
            if '.' in barras_conectadas[0] or '.' in barras_conectadas[1]:
                barras_conectadas[0] = barras_conectadas[0].split('.')[0]
                barras_conectadas[1] = barras_conectadas[1].split('.')[0]
                
            no1 = self.nodes[f"{barras_conectadas[0]}.{1}"]
            no2 = self.nodes[f"{barras_conectadas[1]}.{1}"]
            
            if basekv1 > basekv2:
                n = basekv1 / basekv2
                Ybus[no1:no1+num_phases, no2:no2+num_phases] = (Ybus[no1:no1+num_phases, no2:no2+num_phases])/n
                Ybus[no2:no2+num_phases, no1:no1+num_phases] = (Ybus[no2:no2+num_phases, no1:no1+num_phases])*n
            else:
                n = basekv2 / basekv1
                Ybus[no1:no1+num_phases, no2:no2+num_phases] = (Ybus[no1:no1+num_phases, no2:no2+num_phases])*n
                Ybus[no2:no2+num_phases, no1:no1+num_phases] = (Ybus[no2:no2+num_phases, no1:no1+num_phases])/n
                
            self.DSSCircuit.Transformers.Next

        self.DSSCircuit.Loads.First
        for _ in range(self.DSSCircuit.Loads.Count):
            self.DSSCircuit.SetActiveElement(self.DSSCircuit.Loads.Name)
            Yprim = self.DSSCircuit.ActiveCktElement.Yprim
            real = Yprim[::2]
            imag = Yprim[1::2]*1j
            Yprim = real+imag
            barra_correspondente = self.DSSCircuit.ActiveCktElement.BusNames[0].split('.')[0]
            self.DSSCircuit.SetActiveBus(barra_correspondente)
            fases_barra = self.DSSCircuit.ActiveBus.Nodes
            fases = self.DSSCircuit.ActiveCktElement.NodeOrder - 1
            Yprim = np.array(Yprim, dtype=np.complex128)
            Yprim = self.forma_matriz(fases, fases_barra, Yprim)
            no1 = self.nodes[f"{barra_correspondente}.{min(fases_barra)}"]
            Ybus[no1:no1+len(fases_barra), no1:no1+len(fases_barra)] -= Yprim[:len(fases_barra), :len(fases_barra)]
            self.DSSCircuit.Loads.Next
            
        self.DSSCircuit.Generators.First
        for _ in range(self.DSSCircuit.Generators.Count):
            self.DSSCircuit.SetActiveElement(self.DSSCircuit.Generators.Name)
            Yprim = self.DSSCircuit.ActiveCktElement.Yprim
            real = Yprim[::2]
            imag = Yprim[1::2]*1j
            Yprim = real+imag
            barra_correspondente = self.DSSCircuit.ActiveCktElement.BusNames[0].split('.')[0]
            self.DSSCircuit.SetActiveBus(barra_correspondente)
            fases_barra = self.DSSCircuit.ActiveBus.Nodes
            fases = self.DSSCircuit.ActiveCktElement.NodeOrder - 1
            Yprim = np.array(Yprim, dtype=np.complex128)
            
            Yprim = self.forma_matriz(fases, fases_barra, Yprim)
                
            no1 = self.nodes[f"{barra_correspondente}.{min(fases_barra)}"]
            Ybus[no1:no1+len(fases_barra), no1:no1+len(fases_barra)] -= Yprim[:len(fases_barra), :len(fases_barra)]
            self.DSSCircuit.Generators.Next
        
        self.DSSCircuit.PVSystems.First
        for _ in range(self.DSSCircuit.PVSystems.Count):
            self.DSSCircuit.SetActiveElement(self.DSSCircuit.PVSystems.Name)
            Yprim = self.DSSCircuit.ActiveCktElement.Yprim
            real = Yprim[::2]
            imag = Yprim[1::2]*1j
            Yprim = real+imag
            barra_correspondente = self.DSSCircuit.ActiveCktElement.BusNames[0].split('.')[0]
            self.DSSCircuit.SetActiveBus(barra_correspondente)
            fases_barra = self.DSSCircuit.ActiveBus.Nodes
            fases = self.DSSCircuit.ActiveCktElement.NodeOrder - 1
            Yprim = np.array(Yprim, dtype=np.complex128)
            
            Yprim = self.forma_matriz(fases, fases_barra, Yprim)
                
            no1 = self.nodes[f"{barra_correspondente}.{min(fases_barra)}"]
            Ybus[no1:no1+len(fases_barra), no1:no1+len(fases_barra)] -= Yprim[:len(fases_barra), :len(fases_barra)]
            self.DSSCircuit.PVSystems.Next
            
        self.DSSCircuit.Reactors.First
        for _ in range(self.DSSCircuit.Reactors.Count):
            self.DSSCircuit.SetActiveElement(self.DSSCircuit.Reactors.Name)
            Yprim = self.DSSCircuit.ActiveCktElement.Yprim
            real = Yprim[::2]
            imag = Yprim[1::2]*1j
            Yprim = real+imag
            barra_correspondente = self.DSSCircuit.ActiveCktElement.BusNames[0].split('.')[0]
            self.DSSCircuit.SetActiveBus(barra_correspondente)
            fases_barra = self.DSSCircuit.ActiveBus.Nodes
            fases = self.DSSCircuit.ActiveCktElement.NodeOrder - 1
            Yprim = np.array(Yprim, dtype=np.complex128)
            
            Yprim = self.forma_matriz(fases, fases_barra, Yprim)
                
            no1 = self.nodes[f"{barra_correspondente}.{min(fases_barra)}"]
            Ybus[no1:no1+len(fases_barra), no1:no1+len(fases_barra)] -= Yprim[:len(fases_barra), :len(fases_barra)]
            self.DSSCircuit.Reactors.Next
        
        self.DSSCircuit.SetActiveElement('Vsource.source')
        Yprim = self.DSSCircuit.ActiveCktElement.Yprim
        real = Yprim[::2]
        imag = Yprim[1::2]*1j
        Yprim = real+imag
        Yprim = np.reshape(Yprim, (6, 6))
        Ybus[:3, :3] -= Yprim[:3, :3]
        
        basesY = np.array(self.baseva / ((self.barras['Bases']*1000) ** 2))
        basesY = np.concatenate([[basesY[-1]], basesY[:-1]])
        
        linha = 0
        for baseY, fases in zip(basesY, self.barras['Fases']):
            for fase in fases:
                Ybus[linha] = Ybus[linha] / baseY
                linha += 1
        
        return Ybus

    def Calcula_pesos_anual(self, hora:int) -> tuple:
        matriz_pesos_anual = {}
        dp_anual = {}
        medidas_anual = {}
        #Função auxiliar para inicializar os dicionários que armazenarão os resultados
        def processar_coluna(coluna_dict):
            result_dict = {}
            for key, subdict in coluna_dict.items():
                for hora_key, valores in subdict.items():
                    if hora_key not in result_dict:
                        result_dict[hora_key] = {}
                    result_dict[hora_key][key] = valores
            return result_dict

        inj_pot_at_dict = processar_coluna(self.barras_anuais['Inj_pot_at'].to_dict())
        inj_pot_rat_dict = processar_coluna(self.barras_anuais['Inj_pot_rat'].to_dict())
        tensoes_dict = processar_coluna(self.barras_anuais['Tensao'].to_dict())

        inj_pot_at_list = []
        inj_pot_rat_list = []
        tensoes_list = []
        
        for fases, pot_at, pot_rat, tensao in zip(self.barras_anuais['Fases'], inj_pot_at_dict[f"hora_{hora}"].values(), inj_pot_rat_dict[f"hora_{hora}"].values(), tensoes_dict[f"hora_{hora}"].values()):
                for fase in fases:
                    inj_pot_at_list.append(pot_at[fase])
                    inj_pot_rat_list.append(pot_rat[fase])
                    tensoes_list.append(tensao[fase])
            
        medidas = np.concatenate([inj_pot_at_list[:-3], inj_pot_rat_list[:-3], tensoes_list[:-3]])

        dp = (medidas * 0.01) / (3 * 100)
        dp[dp == 0] = 10**-5
        pesos = dp**-2
        pesos[pesos > 10**10] = 10**10
        matriz_pesos = scsp.lil_matrix((len(pesos), len(pesos)))
        matriz_pesos.setdiag(pesos)
        matriz_pesos_anual[f'hora_{hora}'] = (scsp.csr_matrix(matriz_pesos))
        dp_anual[f'hora_{hora}'] = np.abs(dp)
        medidas_anual[f'hora_{hora}'] = medidas

        return matriz_pesos_anual, dp_anual, medidas_anual
    
    def Calcula_Residuo_anual(self, hora:int):

        #Função auxiliar para inicializar os dicionários que armazenarão os resultados
        def processar_coluna(coluna_dict):
            result_dict = {}
            for key, subdict in coluna_dict.items():
                for hora_key, valores in subdict.items():
                    if hora_key not in result_dict:
                        result_dict[hora_key] = {}
                    result_dict[hora_key][key] = valores
            return result_dict

        tensoes_dict = processar_coluna(self.barras_anuais['Tensao'].to_dict())
        residuo_dict = {}
        self.inj_pot_at_est_dict = {}
        self.inj_pot_rat_est_dict = {}

        angs = self.vet_estados_anuais[f"hora_{hora}"][:len(self.fases[:-(self.count)*3])]
        tensoes = self.vet_estados_anuais[f"hora_{hora}"][len(self.fases[:-(self.count)*3]):]
        ang_ref = np.array([0, -120*2*np.pi / 360,  120*2*np.pi / 360])
        tensoes_dict_valores = [*tensoes_dict[f"hora_{hora}"].values()]
        tensoes_ref = tensoes_dict_valores[self.DSSCircuit.NumBuses-1][:3]
        angs = np.concatenate((ang_ref, angs))
        tensoes = np.concatenate((tensoes_ref, tensoes))
            
        residuo = Residuo(self.barras_anuais, tensoes, angs, anual = True, hora = hora)
            
        residuos_dict = residuo.calc_res_anual(self.Gs, self.Bs, hora, residuo_dict)
            
        self.inj_pot_at_est_dict[f"hora_{hora}"] = np.array(residuo.inj_pot_at_est)
        self.inj_pot_rat_est_dict[f"hora_{hora}"] = np.array(residuo.inj_pot_rat_est)

        return residuos_dict
    
    def Calcula_Jacobiana_anual(self, hora:int) -> dict:

        #Função auxiliar para inicializar os dicionários que armazenarão os resultados
        def processar_coluna(coluna_dict):
            result_dict = {}
            for key, subdict in coluna_dict.items():
                for hora_key, valores in subdict.items():
                    if hora_key not in result_dict:
                        result_dict[hora_key] = {}
                    result_dict[hora_key][key] = valores
            return result_dict

        tensoes_dict = processar_coluna(self.barras_anuais['Tensao'].to_dict())
        jacobiana_dict = {}

        angs = self.vet_estados_anuais[f"hora_{hora}"][:len(self.fases[:-(self.count)*3])]
        tensoes = self.vet_estados_anuais[f"hora_{hora}"][len(self.fases[:-(self.count)*3]):]
        ang_ref = np.array([0, -120*2*np.pi / 360,  120*2*np.pi / 360])
        tensoes_dict_valores = [*tensoes_dict[f"hora_{hora}"].values()]
        tensoes_ref = tensoes_dict_valores[self.DSSCircuit.NumBuses-1][:3]
        angs = np.concatenate((angs, ang_ref))
        tensoes = np.concatenate((tensoes, tensoes_ref))

        jac_anual = Jacobiana(tensoes, angs, self.fases, anual = True, hora = hora)
        jacobiana_dict = jac_anual.derivadas_anual(self.Gs, self.Bs, 
                                                    self.inj_pot_at_est_dict[f"hora_{hora}"], self.inj_pot_rat_est_dict[f"hora_{hora}"], hora, jacobiana_dict)

        return jacobiana_dict

    def progressBar(self, iterable, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        """
        Call in a loop to create terminal progress bar with estimated time remaining.
        @params:
            iterable    - Required  : iterable object (Iterable)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        total = len(iterable)
        start_time = time.time()

        def printProgressBar(iteration):
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / (iteration + 1) * total
            time_remaining = estimated_total_time - elapsed_time
            
            time_format = lambda t: time.strftime("%H:%M:%S", time.gmtime(t))
            
            print(f'\r{prefix} |{bar}| {percent}% {suffix} | Time Remaining: {time_format(time_remaining)}', end=printEnd)

        # Initial Call
        printProgressBar(0)
        
        # Update Progress Bar
        for i, item in enumerate(iterable):
            yield item
            printProgressBar(i + 1)
        
        # Print New Line on Complete
        print()
    
    def run_anual(self, max_error: float, max_iter: int, total_horas: int):

        delx_dict = {}
        matriz_ganho_dict ={}
        seinao_dict = {}

        tempo_residuo = 0
        tempo_jacobiana = 0
        tempo_peso = 0
        tempo_atualizar = 0

        for hora in range(total_horas):
            delx_dict[f"hora_{hora}"] = 1

        '''
        # Definir o tamanho do chunk (por exemplo, uma semana)
        chunk_size = 12  # 7 dias
        
        # Dividir o ano em chunks
        chunks = [(i, min(i + chunk_size, total_horas)) for i in range(0, total_horas, chunk_size)]
        '''

        def run_por_hora(hora_inicial, hora_final,vet_estados_chunk):

            for hora in self.progressBar(range(hora_inicial, hora_final), prefix='Progress:', suffix='Complete', length=50):
                matriz_pesos_anual, dp_anual, medidas_anual = self.Calcula_pesos_anual(hora)
                k = 0

                while(np.max(np.abs(delx_dict[f"hora_{hora}"])) > max_error and max_iter > k):
                    inicio = time.time()
                    residuo_anual = self.Calcula_Residuo_anual(hora)
                    fim_res_anual = time.time()

                    jacobiana_anual = self.Calcula_Jacobiana_anual(hora)
                    fim_jac_anual = time.time()

                    matriz_pesos_anual, dp_anual, medidas_anual = self.Calcula_pesos_anual(hora)
                    fim_pesos_anual = time.time()
                    
                    #Calcula a matriz ganho
                    matriz_ganho_dict[f"hora_{hora}"] = jacobiana_anual[f"hora_{hora}"].T @ matriz_pesos_anual[f"hora_{hora}"] @ jacobiana_anual[f"hora_{hora}"]
                                
                    #Calcula o outro lado da Equação normal
                    seinao_dict[f"hora_{hora}"] = jacobiana_anual[f"hora_{hora}"].T @ matriz_pesos_anual[f"hora_{hora}"] @ residuo_anual[f"hora_{hora}"]

                    delx_dict[f"hora_{hora}"] = np.linalg.inv(matriz_ganho_dict[f"hora_{hora}"].toarray()) @ seinao_dict[f"hora_{hora}"]
                                
                    #Atualiza o vetor de estados
                    vet_estados_chunk[f"hora_{hora}"] = np.add(delx_dict[f"hora_{hora}"], vet_estados_chunk[f"hora_{hora}"])
                    
                    fim = time.time()
                    
                    k += 1
                        
                if self.verbose:
                    print(f'Os resíduos da iteração {k} levaram {tempo_residuo:.3f}s')
                    print(f'A jacobiana da iteração {k} levou {tempo_jacobiana:.3f}s')
                    print(f'Os pesos da iteração {k} levaram {tempo_peso:.3f}s')
                    print(f'Atualizar o vetor de estados da iteração {k} levou {tempo_atualizar:.3f}')
                    print(f'A iteração {k} levou {fim-inicio:.3f}s')

            return vet_estados_chunk

        self.vet_estados_anuais = run_por_hora(self.hora_inicial, self.hora_final, self.vet_estados_anuais)

        return self.vet_estados_anuais