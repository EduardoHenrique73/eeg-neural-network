#!/usr/bin/env python3
"""
Suite de testes consolidada para o sistema EEG
"""

import os
import numpy as np
from dinamica_simbolica import calcular_entropia_shannon, aplicar_dinamica_simbolica
from config import config

def testar_entropia_shannon():
    """Testa a função de entropia de Shannon com diferentes cenários"""
    
    print("🧪 TESTANDO ENTROPIA DE SHANNON")
    print("=" * 50)
    
    # Teste 1: Distribuição uniforme (entropia máxima)
    print("\n1️⃣ Teste: Distribuição Uniforme")
    freq_uniforme = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    entropia_uniforme = calcular_entropia_shannon(freq_uniforme)
    print(f"   Frequências: {freq_uniforme}")
    print(f"   Entropia: {entropia_uniforme:.4f}")
    print(f"   Esperado: ~1.0000 (máximo normalizado)")
    
    # Teste 2: Distribuição concentrada (entropia baixa)
    print("\n2️⃣ Teste: Distribuição Concentrada")
    freq_concentrada = {0: 0.9, 1: 0.05, 2: 0.03, 3: 0.02}
    entropia_concentrada = calcular_entropia_shannon(freq_concentrada)
    print(f"   Frequências: {freq_concentrada}")
    print(f"   Entropia: {entropia_concentrada:.4f}")
    print(f"   Esperado: ~0.3 (baixa entropia)")
    
    # Teste 3: Caso extremo - apenas um símbolo
    print("\n3️⃣ Teste: Apenas um Símbolo")
    freq_unico = {0: 1.0}
    entropia_unico = calcular_entropia_shannon(freq_unico)
    print(f"   Frequências: {freq_unico}")
    print(f"   Entropia: {entropia_unico:.4f}")
    print(f"   Esperado: 0.0000 (entropia mínima)")
    
    # Validação
    print("\n📊 VALIDAÇÃO:")
    if abs(entropia_uniforme - 1.0) < 0.1:
        print("   ✅ Entropia uniforme CORRETA!")
    else:
        print("   ❌ Entropia uniforme INCORRETA!")
    
    if entropia_unico == 0.0:
        print("   ✅ Entropia mínima CORRETA!")
    else:
        print("   ❌ Entropia mínima INCORRETA!")

def criar_arquivo_teste():
    """Cria um arquivo de teste com dados EEG simulados"""
    
    print("\n🧪 CRIANDO ARQUIVO DE TESTE")
    print("=" * 50)
    
    # Gerar dados EEG simulados (1000 amostras)
    np.random.seed(42)  # Para reprodutibilidade
    
    # Sinal base com algumas características EEG
    t = np.linspace(0, 10, 1000)
    sinal_base = np.sin(2 * np.pi * 10 * t)  # Componente de 10 Hz
    sinal_base += 0.5 * np.sin(2 * np.pi * 20 * t)  # Componente de 20 Hz
    sinal_base += 0.3 * np.sin(2 * np.pi * 5 * t)   # Componente de 5 Hz
    
    # Adicionar ruído
    ruido = np.random.normal(0, 0.1, 1000)
    sinal_final = sinal_base + ruido
    
    # Salvar arquivo
    nome_arquivo = "teste_eeg_upload.txt"
    with open(nome_arquivo, 'w') as f:
        for valor in sinal_final:
            f.write(f"{valor:.6f}\n")
    
    print(f"✅ Arquivo de teste criado: {nome_arquivo}")
    print(f"📊 Total de amostras: {len(sinal_final)}")
    print(f"📈 Valores: min={sinal_final.min():.3f}, max={sinal_final.max():.3f}, média={sinal_final.mean():.3f}")
    
    return nome_arquivo

def testar_conexao_banco():
    """Testa a conexão com o banco de dados"""
    
    print("\n🔍 TESTANDO CONEXÃO COM BANCO")
    print("=" * 50)
    
    try:
        import psycopg2
        conexao = psycopg2.connect(**config.get_db_connection_string())
        cursor = conexao.cursor()
        
        # Teste básico
        cursor.execute("SELECT COUNT(*) FROM sinais")
        total_sinais = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM usuarios")
        total_usuarios = cursor.fetchone()[0]
        
        cursor.close()
        conexao.close()
        
        print(f"✅ Conexão com banco OK")
        print(f"   Total de sinais: {total_sinais}")
        print(f"   Total de usuários: {total_usuarios}")
        
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")

def testar_dinamica_simbolica():
    """Testa a dinâmica simbólica com dados simulados"""
    
    print("\n🧮 TESTANDO DINÂMICA SIMBÓLICA")
    print("=" * 50)
    
    # Criar dados de teste
    dados_teste = np.random.normal(0, 1, 1000)
    
    # Salvar temporariamente
    arquivo_temp = "temp_teste.txt"
    with open(arquivo_temp, 'w') as f:
        for valor in dados_teste:
            f.write(f"{valor:.6f}\n")
    
    try:
        # Inserir no banco para teste
        import psycopg2
        conexao = psycopg2.connect(**config.get_db_connection_string())
        cursor = conexao.cursor()
        
        # Criar usuário de teste
        cursor.execute("INSERT INTO usuarios (possui) VALUES (%s) RETURNING id", ('U',))
        id_usuario = cursor.fetchone()[0]
        
        # Criar sinal de teste
        cursor.execute("INSERT INTO sinais (nome, idusuario) VALUES (%s, %s) RETURNING id", 
                      ("teste_ds", id_usuario))
        id_sinal = cursor.fetchone()[0]
        
        # Inserir valores
        valores_para_inserir = [(id_sinal, float(valor)) for valor in dados_teste]
        cursor.executemany("INSERT INTO valores_sinais (idsinal, valor) VALUES (%s, %s)", 
                          valores_para_inserir)
        
        conexao.commit()
        cursor.close()
        conexao.close()
        
        # Testar dinâmica simbólica
        resultado = aplicar_dinamica_simbolica(id_sinal, m=3)
        
        if resultado:
            print(f"✅ Dinâmica simbólica OK")
            print(f"   Entropia: {resultado.get('entropia', 0):.4f}")
            print(f"   Limiar: {resultado.get('limiar', 0):.4f}")
            print(f"   Sequência: {len(resultado.get('sequencia_binaria', []))} elementos")
        else:
            print("❌ Erro na dinâmica simbólica")
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
    
    finally:
        # Limpar arquivo temporário
        if os.path.exists(arquivo_temp):
            os.remove(arquivo_temp)

def executar_todos_testes():
    """Executa todos os testes"""
    
    print("🚀 EXECUTANDO SUITE DE TESTES")
    print("=" * 60)
    
    # Testes
    testar_entropia_shannon()
    testar_conexao_banco()
    criar_arquivo_teste()
    testar_dinamica_simbolica()
    
    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES CONCLUÍDOS!")
    print("\n📋 PRÓXIMOS PASSOS:")
    print("1. Execute: python app.py")
    print("2. Acesse: http://localhost:5000/upload")
    print("3. Use o arquivo: teste_eeg_upload.txt")
    print("4. Verifique os resultados")

if __name__ == "__main__":
    executar_todos_testes()
