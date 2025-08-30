#!/usr/bin/env python3
"""
Teste do MLP Tabular vs CNN para dados tabulares
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_classifier import EEGClassifier
import numpy as np

def testar_modelos():
    """Testa diferentes modelos para comparar performance"""
    
    print("🧪 TESTE DE MODELOS PARA DADOS TABULARES")
    print("=" * 50)
    
    # Criar dataset balanceado
    classifier = EEGClassifier()
    X, y, feature_names = classifier.criar_dataset(limite=40)  # Mais dados para ter ambas as classes
    
    if X is None:
        print("❌ Não foi possível criar dataset")
        return
    
    print(f"📊 Dataset: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"📊 Distribuição: {np.bincount(y)}")
    print(f"📊 Features: {feature_names}")
    print()
    
    # Testar diferentes modelos
    modelos = [
        ('random_forest', 'Random Forest'),
        ('mlp_tabular', 'MLP Tabular'),
        ('cnn', 'CNN 1D'),
        ('lstm', 'LSTM')
    ]
    
    resultados = {}
    
    for tipo_modelo, nome in modelos:
        print(f"🔬 Testando {nome}...")
        
        try:
            # Criar e treinar modelo
            model = EEGClassifier()
            model.criar_modelo(tipo_modelo)
            model.treinar_modelo(X, y, validation_split=0.3)
            
            if model.is_trained:
                # Fazer algumas predições de teste
                predicoes = []
                for i in range(min(5, len(X))):
                    pred = model.prever_sinal(i + 1)  # IDs começam em 1
                    if pred:
                        predicoes.append(pred['probabilidade'])
                
                resultados[nome] = {
                    'treinado': True,
                    'predicoes_teste': predicoes,
                    'media_prob': np.mean(predicoes) if predicoes else 0
                }
                
                print(f"   ✅ {nome} treinado com sucesso")
                print(f"   📊 Predições de teste: {predicoes}")
                print(f"   📊 Média das probabilidades: {np.mean(predicoes):.3f}")
            else:
                resultados[nome] = {'treinado': False}
                print(f"   ❌ {nome} falhou no treinamento")
                
        except Exception as e:
            resultados[nome] = {'treinado': False, 'erro': str(e)}
            print(f"   ❌ Erro no {nome}: {e}")
        
        print()
    
    # Resumo dos resultados
    print("📋 RESUMO DOS RESULTADOS")
    print("=" * 30)
    
    for nome, resultado in resultados.items():
        if resultado['treinado']:
            print(f"✅ {nome}: Funcionando (prob média: {resultado['media_prob']:.3f})")
        else:
            erro = resultado.get('erro', 'Falha no treinamento')
            print(f"❌ {nome}: {erro}")
    
    print()
    print("💡 RECOMENDAÇÕES:")
    print("- Para dados tabulares (19 features estatísticas):")
    print("  • Random Forest: Melhor para datasets pequenos")
    print("  • MLP Tabular: Boa alternativa neural")
    print("  • CNN/LSTM: Melhor para séries temporais brutas")
    print("- Para usar CNN/LSTM efetivamente, use janelas do sinal EEG bruto")

if __name__ == "__main__":
    testar_modelos()
