import os
import numpy as np
from collections import Counter
import psycopg2

# Configuração robusta do Matplotlib para ambiente de servidor
import matplotlib
matplotlib.use('Agg')  # Backend não interativo para servidores
matplotlib.rcParams['figure.max_open_warning'] = 0  # Desabilitar avisos de figuras
matplotlib.rcParams['figure.dpi'] = 100  # DPI mais baixo para melhor performance
matplotlib.rcParams['savefig.dpi'] = 100
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.1

import matplotlib.pyplot as plt
plt.ioff()  # Desabilitar modo interativo

# Configurar para evitar problemas de renderer
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

from config import config

def obter_conexao_db():
    """Conecta ao banco de dados PostgreSQL"""
    return psycopg2.connect(**config.get_db_connection_string())

def obter_dados_sinal(cursor, id_sinal):
    """Busca os valores brutos do sinal no banco de dados"""
    cursor.execute("SELECT valor FROM valores_sinais WHERE idsinal = %s", (id_sinal,))
    return np.array([row[0] for row in cursor.fetchall()])

def verificar_dados(dados):
    """Valida se todos os dados são numéricos"""
    if not all(isinstance(x, (int, float)) for x in dados):
        raise ValueError("Os dados contêm valores não numéricos.")

def obter_limiar_sql(cursor, id_sinal):
    """Calcula a média do sinal diretamente no banco"""
    cursor.execute("SELECT AVG(valor) FROM valores_sinais WHERE idsinal = %s", (id_sinal,))
    return cursor.fetchone()[0] or 0.0

def gerar_sequencia_binaria(sinal, limiar):
    """Gera a sequência de 0s e 1s baseada na média"""
    return ['1' if x >= limiar else '0' for x in sinal]

def gerar_grupos_deslizantes(sequencia, m=3):
    """Cria janelas deslizantes de tamanho m"""
    return [''.join(sequencia[i:i+m]) for i in range(len(sequencia)-m+1)]

def converter_para_decimal(grupos):
    """Converte os grupos binários para valores decimais"""
    if not grupos:
        return []
    return [int(grupo, 2) for grupo in grupos]

def calcular_frequencia(palavras_decimais):
    """Calcula a frequência relativa dos valores"""
    if not palavras_decimais:
        return {}
    
    contagem = Counter(palavras_decimais)
    total = sum(contagem.values())
    return {k: v/total for k, v in contagem.items()}

def calcular_entropia_shannon(frequencias):
    """
    Calcula a entropia de Shannon normalizada usando logaritmo natural (ln)
    - Usa logaritmo natural (ln)
    - Retorna valor normalizado entre 0 e 1
    - Ignora valores de frequência relativa que sejam 0 ou 1
    """
    if not frequencias:
        return 0.0
    
    probabilidades = np.array(list(frequencias.values()))
    probabilidades_filtradas = probabilidades[(probabilidades > 0) & (probabilidades < 1)]
    if len(probabilidades_filtradas) == 0:
        return 0.0
    probabilidades_norm = probabilidades_filtradas / np.sum(probabilidades_filtradas)
    entropia_bruta = -np.sum(probabilidades_norm * np.log(probabilidades_norm))
    n_simbolos = len(probabilidades_filtradas)
    if n_simbolos > 1:
        entropia_maxima = np.log(n_simbolos)
        entropia_normalizada = entropia_bruta / entropia_maxima
    else:
        entropia_normalizada = 0.0
    return max(0.0, min(1.0, entropia_normalizada))

def plotar_histograma(frequencias, nome_base):
    """Gera e salva o histograma com rótulos binários"""
    try:
        chaves = sorted(frequencias.keys())
        
        # Criar figura com configurações específicas
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(
            range(len(chaves)),
            [frequencias[k] for k in chaves],
            tick_label=[f"{k:03b}" for k in chaves]  # Formato binário de 3 bits
        )

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')

        ax.set_xlabel("Grupos Binários (3 bits)", fontsize=12)
        ax.set_ylabel("Frequência Relativa", fontsize=12)
        ax.set_title(f"Distribuição de Padrões - {nome_base}", fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Salvar figura
        nome_arquivo = f"{nome_base}_histograma.png"
        caminho = os.path.join("static", nome_arquivo)
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        
        fig.savefig(caminho, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        return caminho
        
    except Exception as e:
        print(f"Erro ao plotar histograma para {nome_base}: {e}")
        # Retornar caminho vazio em caso de erro
        return ""

def plotar_sequencia_binaria(sequencia, nome_base):
    """Plota a sequência binária ao longo do tempo"""
    try:
        # Criar figura com configurações específicas
        fig, ax = plt.subplots(figsize=(12, 4))

        tempo = np.arange(len(sequencia))
        ax.step(tempo, [int(b) for b in sequencia], 
                where='post', color='#1f77b4', linewidth=1.5)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0 (Abaixo da média)', '1 (Acima da média)'])
        ax.set_xlabel('Tempo (amostras)', fontsize=12)
        ax.set_ylabel('Estado Binário', fontsize=12)
        ax.set_title(f'Sequência Binária - {nome_base}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(sequencia))

        # Salvar figura
        nome_arquivo = f"{nome_base}_sequencia.png"
        caminho = os.path.join("static", nome_arquivo)
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        
        fig.savefig(caminho, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        return caminho
        
    except Exception as e:
        print(f"Erro ao plotar sequência para {nome_base}: {e}")
        # Retornar caminho vazio em caso de erro
        return ""

def aplicar_dinamica_simbolica(id_sinal, m=3):
    """Função principal que orquestra todo o processo"""
    conexao = None
    cursor = None
    
    try:
        conexao = obter_conexao_db()
        cursor = conexao.cursor()

        dados_brutos = obter_dados_sinal(cursor, id_sinal)
        verificar_dados(dados_brutos)
        limiar = obter_limiar_sql(cursor, id_sinal)
        sequencia_binaria = gerar_sequencia_binaria(dados_brutos, limiar)
        grupos_binarios = gerar_grupos_deslizantes(sequencia_binaria, m)
        palavras_decimais = converter_para_decimal(grupos_binarios)
        frequencias = calcular_frequencia(palavras_decimais)
        
        # Calcula a entropia de Shannon
        entropia = calcular_entropia_shannon(frequencias)

        nome_base = f"sinal_{id_sinal}"
        
        # Tentar gerar gráficos com tratamento de erro
        try:
            caminho_histograma = plotar_histograma(frequencias, nome_base)
        except Exception as e:
            print(f"Erro ao gerar histograma para sinal {id_sinal}: {e}")
            caminho_histograma = ""
            
        try:
            caminho_sequencia = plotar_sequencia_binaria(sequencia_binaria, nome_base)
        except Exception as e:
            print(f"Erro ao gerar sequência para sinal {id_sinal}: {e}")
            caminho_sequencia = ""

        return {
            'caminho_histograma': caminho_histograma,
            'caminho_sequencia': caminho_sequencia,
            'sequencia_binaria': sequencia_binaria,
            'grupos_binarios': grupos_binarios,
            'palavras_decimais': palavras_decimais,
            'limiar': limiar,
            'entropia': entropia,
            'frequencias': frequencias
        }

    except Exception as e:
        print(f"Erro geral na dinâmica simbólica para sinal {id_sinal}: {e}")
        return None
        
    finally:
        if cursor:
            cursor.close()
        if conexao:
            conexao.close()
