#!/usr/bin/env python3
"""
Script simples para testar os modelos CNN e LSTM
"""

import numpy as np
from ml_classifier import EEGClassifier

def testar_modelos():
    """Testa se os modelos CNN e LSTM estão funcionando"""
    
    print("🧪 Testando modelos CNN e LSTM...")
    
    # Testar CNN
    try:
        print("\n📊 Testando CNN...")
        classifier_cnn = EEGClassifier()
        classifier_cnn.criar_modelo('cnn')
        
        # Criar dataset
        X, y, _ = classifier_cnn.criar_dataset(limite=10)
        if X is not None and len(X) > 0:
            print(f"✅ Dataset criado: {X.shape}")
            
            # Treinar
            classifier_cnn.treinar_modelo(X, y)
            print("✅ CNN treinado!")
            
            # Testar predição
            if classifier_cnn.is_trained:
                # Simular dados de teste
                X_teste = np.random.random((1, X.shape[1]))
                predicao = classifier_cnn.model.predict(X_teste)
                print(f"✅ Predição CNN: {predicao}")
            else:
                print("❌ CNN não está treinado")
        else:
            print("❌ Não foi possível criar dataset")
            
    except Exception as e:
        print(f"❌ Erro no CNN: {e}")
        import traceback
        traceback.print_exc()
    
    # Testar LSTM
    try:
        print("\n📊 Testando LSTM...")
        classifier_lstm = EEGClassifier()
        classifier_lstm.criar_modelo('lstm')
        
        # Criar dataset
        X, y, _ = classifier_lstm.criar_dataset(limite=10)
        if X is not None and len(X) > 0:
            print(f"✅ Dataset criado: {X.shape}")
            
            # Treinar
            classifier_lstm.treinar_modelo(X, y)
            print("✅ LSTM treinado!")
            
            # Testar predição
            if classifier_lstm.is_trained:
                # Simular dados de teste
                X_teste = np.random.random((1, X.shape[1]))
                predicao = classifier_lstm.model.predict(X_teste)
                print(f"✅ Predição LSTM: {predicao}")
            else:
                print("❌ LSTM não está treinado")
        else:
            print("❌ Não foi possível criar dataset")
            
    except Exception as e:
        print(f"❌ Erro no LSTM: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    testar_modelos()
