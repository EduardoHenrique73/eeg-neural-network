#!/usr/bin/env python3
"""
Script para testar se o problema do redimensionamento foi corrigido
"""

import numpy as np
from ml_classifier import EEGClassifier

def testar_keras_corrigido():
    """Testa se o problema do redimensionamento foi corrigido"""
    
    print("🧪 Testando CNN com redimensionamento corrigido...")
    
    try:
        # Criar classificador
        classifier = EEGClassifier()
        
        # Criar dataset
        X, y, _ = classifier.criar_dataset(limite=10)
        if X is not None and len(X) > 0:
            print(f"✅ Dataset criado: {X.shape}")
            
            # Criar e treinar CNN
            classifier.criar_modelo('cnn')
            classifier.treinar_modelo(X, y)
            
            if classifier.is_trained:
                print("✅ CNN treinado com sucesso!")
                
                # Testar predição
                resultado = classifier.prever_sinal(1)
                if resultado:
                    print(f"✅ Predição CNN: {resultado}")
                else:
                    print("❌ Falha na predição")
            else:
                print("❌ CNN não está treinado")
        else:
            print("❌ Não foi possível criar dataset")
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    testar_keras_corrigido()
