#!/usr/bin/env python3
"""
Script para verificar onde estão os arquivos das IAs
"""

import os
import glob

def verificar_arquivos_ias():
    """Verifica onde estão os arquivos das IAs"""
    
    print("🔍 Verificando arquivos das IAs...")
    print("=" * 50)
    
    # Verificar arquivos de modelo
    arquivos_modelo = [
        'modelo_eeg.pkl',      # Random Forest
        'modelo_cnn.pkl',      # CNN
        'modelo_lstm.pkl'      # LSTM
    ]
    
    for arquivo in arquivos_modelo:
        if os.path.exists(arquivo):
            tamanho = os.path.getsize(arquivo) / 1024  # KB
            print(f"✅ {arquivo}: {tamanho:.1f} KB")
        else:
            print(f"❌ {arquivo}: Não encontrado")
    
    print("\n📁 Todos os arquivos .pkl no diretório:")
    arquivos_pkl = glob.glob("*.pkl")
    for arquivo in arquivos_pkl:
        tamanho = os.path.getsize(arquivo) / 1024  # KB
        print(f"   {arquivo}: {tamanho:.1f} KB")
    
    print("\n📊 Resumo:")
    if os.path.exists('modelo_eeg.pkl'):
        print("   ✅ Random Forest: Salvo")
    else:
        print("   ❌ Random Forest: Não salvo")
    
    if os.path.exists('modelo_cnn.pkl'):
        print("   ✅ CNN: Salvo")
    else:
        print("   ❌ CNN: Não salvo")
    
    if os.path.exists('modelo_lstm.pkl'):
        print("   ✅ LSTM: Salvo")
    else:
        print("   ❌ LSTM: Não salvo")

if __name__ == "__main__":
    verificar_arquivos_ias()

