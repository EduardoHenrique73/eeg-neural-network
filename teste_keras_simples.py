#!/usr/bin/env python3
"""
Teste simples dos modelos Keras
"""

import numpy as np
from ml_classifier import EEGClassifier

def testar_modelos_keras():
    """Testa os modelos Keras com dados simulados"""
    
    print("🧪 TESTE SIMPLES DOS MODELOS KERAS")
    print("="*50)
    
    # Criar dados simulados
    n_samples = 50
    n_features = 19
    
    # Dados de entrada (features extraídas)
    X = np.random.randn(n_samples, n_features)
    
    # Labels (0 ou 1)
    y = np.random.randint(0, 2, n_samples)
    
    print(f"📊 Dados de entrada: {X.shape}")
    print(f"📊 Labels: {y.shape}")
    print(f"📊 Distribuição: {np.bincount(y)}")
    
    # Testar cada modelo
    modelos = [
        ("CNN", "cnn"),
        ("LSTM", "lstm"),
        ("Hybrid", "hybrid")
    ]
    
    for nome, tipo in modelos:
        print(f"\n{'='*30}")
        print(f"🧠 TESTANDO: {nome}")
        print(f"{'='*30}")
        
        try:
            # Criar modelo
            classifier = EEGClassifier()
            classifier.criar_modelo(tipo)
            print(f"✅ Modelo {nome} criado")
            
            # Treinar modelo
            resultado = classifier.treinar_modelo(X, y, validation_split=0.2)
            print(f"✅ Modelo {nome} treinado")
            
            # Fazer predição
            predicao = classifier.prever_sinal(1)  # Simular predição
            print(f"✅ Predição feita: {predicao}")
            
        except Exception as e:
            print(f"❌ Erro no modelo {nome}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    testar_modelos_keras()
