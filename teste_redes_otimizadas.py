#!/usr/bin/env python3
"""
Script para testar as redes neurais otimizadas
"""

import numpy as np
from ml_classifier import EEGClassifier

def testar_redes_otimizadas():
    """Testa as redes neurais otimizadas"""
    
    print("🧪 Testando redes neurais otimizadas...")
    
    # Testar CNN otimizada
    try:
        print("\n📊 Testando CNN otimizada...")
        classifier_cnn = EEGClassifier()
        classifier_cnn.criar_modelo('cnn')
        
        # Criar dataset
        X, y, _ = classifier_cnn.criar_dataset(limite=20)
        if X is not None and len(X) > 0:
            print(f"✅ Dataset criado: {X.shape}")
            
            # Treinar
            classifier_cnn.treinar_modelo(X, y)
            
            if classifier_cnn.is_trained:
                print("✅ CNN otimizada treinada!")
                
                # Testar predições
                for i in range(1, 4):
                    resultado = classifier_cnn.prever_sinal(i)
                    if resultado:
                        confianca = resultado['probabilidade'] * 100
                        print(f"   Sinal {i}: {resultado['classe_predita']} ({confianca:.1f}%)")
            else:
                print("❌ CNN não está treinada")
        else:
            print("❌ Não foi possível criar dataset")
            
    except Exception as e:
        print(f"❌ Erro no CNN: {e}")
        import traceback
        traceback.print_exc()
    
    # Testar LSTM otimizada
    try:
        print("\n📊 Testando LSTM otimizada...")
        classifier_lstm = EEGClassifier()
        classifier_lstm.criar_modelo('lstm')
        
        # Criar dataset
        X, y, _ = classifier_lstm.criar_dataset(limite=20)
        if X is not None and len(X) > 0:
            print(f"✅ Dataset criado: {X.shape}")
            
            # Treinar
            classifier_lstm.treinar_modelo(X, y)
            
            if classifier_lstm.is_trained:
                print("✅ LSTM otimizada treinada!")
                
                # Testar predições
                for i in range(1, 4):
                    resultado = classifier_lstm.prever_sinal(i)
                    if resultado:
                        confianca = resultado['probabilidade'] * 100
                        print(f"   Sinal {i}: {resultado['classe_predita']} ({confianca:.1f}%)")
            else:
                print("❌ LSTM não está treinada")
        else:
            print("❌ Não foi possível criar dataset")
            
    except Exception as e:
        print(f"❌ Erro no LSTM: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    testar_redes_otimizadas()

