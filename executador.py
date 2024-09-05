import threading
import os


def inicia_programa(argumentos):
    os.system('python geracao_dados.py {}'.format(argumentos))
    # Ex: os.system('py -3.7 x.py')

if __name__ == "__main__":

    argumentos = ['0 120','120 241']

    processos = []
    for arquivo in argumentos:
        processos.append(threading.Thread(target=inicia_programa, args=(arquivo,)))
        # Ex: adicionar o porcesso `threading.Thread(target=inicia_programa, args=('x.py',))`

    for processo in processos:
        processo.start()