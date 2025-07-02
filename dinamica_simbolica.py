import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import psycopg2
import matplotlib
matplotlib.use('Agg')  # Backend não interativo para servidores

def obter_conexao_db():
    """Conecta ao banco de dados PostgreSQL"""
    return psycopg2.connect(
        dbname="eeg-projeto",
        user="postgres",
        password="EEG@321",
        host="localhost",
        port="5432"
    )

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

def plotar_histograma(frequencias, nome_base):
    """Gera e salva o histograma com rótulos binários"""
    chaves = sorted(frequencias.keys())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        range(len(chaves)),
        [frequencias[k] for k in chaves],
        tick_label=[f"{k:03b}" for k in chaves]  # Formato binário de 3 bits
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    plt.xlabel("Grupos Binários (3 bits)", fontsize=12)
    plt.ylabel("Frequência Relativa", fontsize=12)
    plt.title(f"Distribuição de Padrões - {nome_base}", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    nome_arquivo = f"{nome_base}_histograma.png"
    caminho = os.path.join("static", nome_arquivo)
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho)
    plt.close()

    return caminho

def plotar_sequencia_binaria(sequencia, nome_base):
    """Plota a sequência binária ao longo do tempo"""
    plt.figure(figsize=(12, 4))

    tempo = np.arange(len(sequencia))
    plt.step(tempo, [int(b) for b in sequencia], 
             where='post', color='#1f77b4', linewidth=1.5)

    plt.yticks([0, 1], ['0 (Abaixo da média)', '1 (Acima da média)'])
    plt.xlabel('Tempo (amostras)', fontsize=12)
    plt.ylabel('Estado Binário', fontsize=12)
    plt.title(f'Sequência Binária - {nome_base}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, len(sequencia))

    nome_arquivo = f"{nome_base}_sequencia.png"
    caminho = os.path.join("static", nome_arquivo)
    plt.savefig(caminho, bbox_inches='tight')
    plt.close()

    return caminho

def aplicar_dinamica_simbolica(id_sinal, m=3):
    """Função principal que orquestra todo o processo"""
    conexao = obter_conexao_db()
    cursor = conexao.cursor()

    try:
        dados_brutos = obter_dados_sinal(cursor, id_sinal)
        verificar_dados(dados_brutos)
        limiar = obter_limiar_sql(cursor, id_sinal)
        sequencia_binaria = gerar_sequencia_binaria(dados_brutos, limiar)
        grupos_binarios = gerar_grupos_deslizantes(sequencia_binaria, m)
        palavras_decimais = converter_para_decimal(grupos_binarios)
        frequencias = calcular_frequencia(palavras_decimais)

        nome_base = f"sinal_{id_sinal}"
        caminho_histograma = plotar_histograma(frequencias, nome_base)
        caminho_sequencia = plotar_sequencia_binaria(sequencia_binaria, nome_base)

        return {
            'caminho_histograma': caminho_histograma,
            'caminho_sequencia': caminho_sequencia,
            'sequencia_binaria': sequencia_binaria,
            'grupos_binarios': grupos_binarios,
            'palavras_decimais': palavras_decimais,
            'limiar': limiar
        }

    finally:
        cursor.close()
        conexao.close()
