import numpy as np
import pandas as pd

class Residuo():
    def __init__(self, barras: pd.DataFrame, tensoes, angs, anual:bool, hora:int) -> None:
        if not anual:
            self.barras = barras
            self.tensoes = tensoes
            self.angs = angs
            self.matriz_tensoes, self.matriz_angs = self.ajustar_entradas(self.tensoes, self.angs)
        else:
            self.hora = hora
            self.barras_anuais = barras
            self.tensoes = tensoes
            self.angs = angs
            self.matriz_tensoes, self.matriz_angs = self.ajustar_entradas(self.tensoes, self.angs)

    def calc_res(self, Gs, Bs) -> np.array:
        diff_angs = self.angs - self.matriz_angs.T
        diff_angs = diff_angs.T
        inj_pot_at = []
        inj_pot_rat = []
        tensoes = []
        for fases, pot_at, pot_rat, tensao in zip(self.barras['Fases'], self.barras['Inj_pot_at'], self.barras['Inj_pot_rat'], self.barras['Tensao']):
            for fase in fases:
                inj_pot_at.append(pot_at[fase])
                inj_pot_rat.append(pot_rat[fase])
                tensoes.append(tensao[fase])
        
        #res_inj_pot_at = self.barras['Inj_pot_at'].to_numpy()
        self.inj_pot_at_est = self.tensoes[3:] * np.sum(self.matriz_tensoes * (Gs * np.cos(diff_angs) + Bs * np.sin(diff_angs)), axis=1)[3:]
        res_inj_pot_at = np.array(inj_pot_at)[:-3] - self.inj_pot_at_est
        
        #res_inj_pot_rat = self.barras['Inj_pot_rat'].to_numpy()
        self.inj_pot_rat_est = self.tensoes[3:] * np.sum(self.matriz_tensoes * (Gs * np.sin(diff_angs) - Bs * np.cos(diff_angs)), axis=1)[3:]
        res_inj_pot_rat = np.array(inj_pot_rat)[:-3] - self.inj_pot_rat_est

        #res_tensao = self.barras['Tensao'].to_numpy()
        res_tensao = np.array(tensoes)[:-3] - self.tensoes[3:]
        
        return np.concatenate([res_inj_pot_at, res_inj_pot_rat, res_tensao])
    
    def calc_res_anual(self, Gs, Bs, hora, residuo_dict) -> dict: 
        #inj_pot_at_est_dict = {}
        #inj_pot_rat_est_dict = {}
        #Função auxiliar para inicializar os dicionários que armazenarão os resultados
        def processar_coluna(coluna_dict):
            result_dict = {}
            for key, subdict in coluna_dict.items():
                for hora_key, valores in subdict.items():
                    if hora_key not in result_dict:
                        result_dict[hora_key] = {}
                    result_dict[hora_key][key] = valores
            return result_dict

        coluna1_dict = self.barras_anuais['Inj_pot_at'].to_dict()
        coluna2_dict = self.barras_anuais['Inj_pot_rat'].to_dict()
        coluna3_dict = self.barras_anuais['Tensao'].to_dict()

        inj_pot_at_dict = processar_coluna(coluna1_dict)
        inj_pot_rat_dict = processar_coluna(coluna2_dict)
        tensoes_dict = processar_coluna(coluna3_dict)

        diff_angs = self.angs - self.matriz_angs.T
        diff_angs = diff_angs.T
        inj_pot_at_list = []
        inj_pot_rat_list = []
        tensoes_list = []
        for fases, pot_at, pot_rat, tensao in zip(self.barras_anuais['Fases'], inj_pot_at_dict[f"hora_{hora}"].values(), inj_pot_rat_dict[f"hora_{hora}"].values(), tensoes_dict[f"hora_{hora}"].values()):
            for fase in fases:
                inj_pot_at_list.append(pot_at[fase])
                inj_pot_rat_list.append(pot_rat[fase])
                tensoes_list.append(tensao[fase])

        self.inj_pot_at_est = self.tensoes[3:] * np.sum(self.matriz_tensoes * (Gs * np.cos(diff_angs) + Bs * np.sin(diff_angs)), axis=1)[3:]
        res_inj_pot_at = np.array(inj_pot_at_list)[:-3] - self.inj_pot_at_est
            
        self.inj_pot_rat_est = self.tensoes[3:] * np.sum(self.matriz_tensoes * (Gs * np.sin(diff_angs) - Bs * np.cos(diff_angs)), axis=1)[3:]
        res_inj_pot_rat = np.array(inj_pot_rat_list)[:-3] - self.inj_pot_rat_est

        res_tensao = np.array(tensoes_list)[:-3] - self.tensoes[3:]

        #self.inj_pot_at_est_dict[f'hora_{hora}'] = self.inj_pot_at_est
        #self.inj_pot_rat_est_dict[f'hora_{hora}'] = self.inj_pot_rat_est
        residuo_dict[f'hora_{hora}'] = np.concatenate([res_inj_pot_at, res_inj_pot_rat, res_tensao])
        return residuo_dict
    
    def ajustar_entradas(self, tensoes: np.ndarray, angs: np.ndarray):
        #Cria matrizes cujas linhas são repetições dos vetores, pois é mais fácil manipular
        matriz_tensoes = np.array([tensoes for _ in range(len(tensoes))])
        matriz_angs = np.array([angs for _ in range(len(angs))])
        
        return matriz_tensoes, matriz_angs