#!/usr/bin/env python3
"""
Script de teste para o sistema de upload de arquivos EEG
"""

import os
import numpy as np

def criar_arquivo_teste():
    """Cria um arquivo de teste com dados EEG simulados"""
    
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

def verificar_arquivo(nome_arquivo):
    """Verifica se o arquivo foi criado corretamente"""
    if os.path.exists(nome_arquivo):
        with open(nome_arquivo, 'r') as f:
            linhas = f.readlines()
        
        valores = []
        for linha in linhas:
            linha = linha.strip()
            if linha and linha.replace('.', '').replace('-', '').isdigit():
                valores.append(float(linha))
        
        print(f"✅ Arquivo verificado: {len(valores)} valores numéricos encontrados")
        return True
    else:
        print(f"❌ Arquivo não encontrado: {nome_arquivo}")
        return False

if __name__ == "__main__":
    print("🧪 Criando arquivo de teste para upload...")
    arquivo_teste = criar_arquivo_teste()
    verificar_arquivo(arquivo_teste)
    
    print("\n📋 Instruções para teste:")
    print("1. Execute: python app.py")
    print("2. Acesse: http://localhost:5000")
    print("3. Clique em 'Escolher Arquivo EEG'")
    print("4. Selecione o arquivo: teste_eeg_upload.txt")
    print("5. Clique em 'Processar'")
    print("6. Aguarde o processamento completo")
    print("7. Verifique os resultados exibidos")

