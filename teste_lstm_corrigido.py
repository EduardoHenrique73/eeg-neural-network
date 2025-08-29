#!/usr/bin/env python3
"""
Script para testar se o LSTM também está funcionando
"""

import numpy as np
from ml_classifier import EEGClassifier

def testar_lstm_corrigido():
    """Testa se o LSTM está funcionando"""
    
    print("🧪 Testando LSTM com redimensionamento corrigido...")
    
    try:
        # Criar classificador
        classifier = EEGClassifier()
        
        # Criar dataset
        X, y, _ = classifier.criar_dataset(limite=10)
        if X is not None and len(X) > 0:
            print(f"✅ Dataset criado: {X.shape}")
            
            # Criar e treinar LSTM
            classifier.criar_modelo('lstm')
            classifier.treinar_modelo(X, y)
            
            if classifier.is_trained:
                print("✅ LSTM treinado com sucesso!")
                
                # Testar predição
                resultado = classifier.prever_sinal(1)
                if resultado:
                    print(f"✅ Predição LSTM: {resultado}")
                else:
                    print("❌ Falha na predição")
            else:
                print("❌ LSTM não está treinado")
        else:
            print("❌ Não foi possível criar dataset")
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    testar_lstm_corrigido()
