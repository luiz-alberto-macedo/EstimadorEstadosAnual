import numpy as np
import pandas as pd

class Residuo():
    def __init__(self, barras: pd.DataFrame) -> None:
        self.vet_inj_at = []
        self.vet_inj_rat = []
        self.vet_tensao = []
        self.barras = barras

    def Residuo_inj_pot_at(self, index_barra: int, fase: int, tensao_estimada: float, tensoes,
                           diff_angulos: np.array, Gs: np.array, Bs: np.array) -> None:
        inj_pot_est = tensao_estimada * np.sum(tensoes * (Gs * np.cos(diff_angulos) + Bs * np.sin(diff_angulos)))
        inj_pot_med = self.barras['Inj_pot_at'][index_barra][fase]
        self.barras['Inj_pot_at_est'][index_barra][fase] = inj_pot_est
        self.vet_inj_at.append(inj_pot_med - inj_pot_est)
        
    def Residuo_inj_pot_rat(self, index_barra: int, fase: int, tensao_estimada: float, tensoes,
                            diff_angulos: np.array, Gs: np.array, Bs: np.array) -> None:
        inj_pot_est = tensao_estimada * np.sum(tensoes * (Gs * np.sin(diff_angulos) - Bs * np.cos(diff_angulos)))
        inj_pot_med = self.barras['Inj_pot_rat'][index_barra][fase]
        self.barras['Inj_pot_rat_est'][index_barra][fase] = inj_pot_est
        self.vet_inj_rat.append(inj_pot_med - inj_pot_est)

    def Residuo_tensao(self, index_barra: int, fase: int, tensao_estimada: float) -> None:
        tensao = self.barras['Tensao'][index_barra][fase]
        self.vet_tensao.append(tensao - tensao_estimada)

    def Residuo_fluxo_pot_at(self, vet_estados: np.array, fases: np.array, residuo_atual: int, index_barra1: int,
                            elemento: str, baseva, barras: pd.DataFrame, DSSCircuit, nodes: dict, Ybus) -> int:
        barra1 = barras['nome_barra'][index_barra1]
        basekv = barras['Bases'][index_barra1]
        baseY = baseva / ((basekv*1000)**2)
        num_buses = DSSCircuit.NumBuses
        Bshmatrix = np.zeros((3, 3))
        barra2 = DSSCircuit.ActiveCktElement.BusNames[1]
        index_barra2 = barras[barras['nome_barra'] == barra2].index.values[0]
        
        tensao_estimada = vet_estados[(DSSCircuit.NumBuses+index_barra1)*3:(DSSCircuit.NumBuses+index_barra1)*3 + 3]
        ang_estimado = vet_estados[(index_barra1*3):(index_barra1*3) + 3]
        
        for fase in fases:
            no1 = nodes[barra1+f'.{fase+1}']

            pot_ativa_estimada = 0
            for m in fases:
                no2 = nodes[barra2+f'.{m+1}']
                Yij = Ybus[no1, no2]
                Gs = np.real(Yij)
                Bs = np.imag(Yij)
                Bsh = Bshmatrix[fase, m]
                tensao_estimada2 = vet_estados[DSSCircuit.NumBuses*3 + (index_barra2*3) + m]
                ang_estimado2 = vet_estados[(index_barra2*3) + m]
                #Calcula a potencia com base nas tensões e ângulos estimados
                parte1 = tensao_estimada[fase]*tensao_estimada[m]*(Gs*np.cos(ang_estimado[fase]-ang_estimado[m])+(Bs+Bsh)*np.sin(ang_estimado[fase]-ang_estimado[m]))
                parte2 = tensao_estimada2*tensao_estimada[fase]*(Gs*np.cos(ang_estimado[fase]-ang_estimado2) + (Bs*np.sin(ang_estimado[fase]-ang_estimado2)))
                pot_ativa_estimada += (parte1 - parte2)
            potencia_at = barras['Flux_pot_at'][index_barra1][0][1][fase]
            self.vetor_residuos.append(potencia_at - pot_ativa_estimada)
            residuo_atual += 1
            
        return residuo_atual
    
    def Residuo_fluxo_pot_rat(self, vet_estados: np.array, fases: np.array, residuo_atual: int, index_barra1: int,
                            elemento: str, baseva, barras: pd.DataFrame, DSSCircuit, nodes: dict, Ybus) -> int:
        barra1 = barras['nome_barra'][index_barra1]
        basekv = barras['Bases'][index_barra1]
        baseY = baseva / ((basekv*1000)**2)
        num_buses = DSSCircuit.NumBuses
        Bshmatrix = np.zeros((3, 3))
        barra2 = DSSCircuit.ActiveCktElement.BusNames[1]
        index_barra2 = barras[barras['nome_barra'] == barra2].index.values[0]
        
        tensao_estimada = vet_estados[(DSSCircuit.NumBuses+index_barra1)*3:(DSSCircuit.NumBuses+index_barra1)*3 + 3]
        ang_estimado = vet_estados[(index_barra1*3):(index_barra1*3) + 3]
        
        for fase in fases:
            no1 = nodes[barra1+f'.{fase+1}']

            pot_reativa_estimada = 0
            for m in fases:
                no2 = nodes[barra2+f'.{m+1}']
                Yij = Ybus[no1, no2]
                Gs = np.real(Yij)
                Bs = np.imag(Yij)
                Bsh = Bshmatrix[fase, m]
                tensao_estimada2 = vet_estados[DSSCircuit.NumBuses*3 + (index_barra2*3) + m]
                ang_estimado2 = vet_estados[(index_barra2*3) + m]
                #Calcula a potencia com base nas tensões e ângulos estimados
                parte1 = tensao_estimada[fase]*tensao_estimada[m]*(Gs*np.sin(ang_estimado[fase]-ang_estimado[m])-(Bs+Bsh)*np.cos(ang_estimado[fase]-ang_estimado[m]))
                parte2 = tensao_estimada2*tensao_estimada[fase]*(Gs*np.sin(ang_estimado[fase]-ang_estimado2) - (Bs*np.cos(ang_estimado[fase]-ang_estimado2)))
                pot_reativa_estimada += (parte1 - parte2)
            potencia_rat = barras['Flux_pot_rat'][index_barra1][0][1][fase]
            self.vetor_residuos.append(potencia_rat - pot_reativa_estimada)
            residuo_atual += 1
            
        return residuo_atual
