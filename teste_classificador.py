#!/usr/bin/env python3
"""
Script de teste para diagnosticar problemas no classificador EEG
"""

import numpy as np
import pandas as pd
from ml_classifier import EEGClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def testar_classificador():
    """Testa o classificador e analisa as probabilidades"""
    print("🔍 DIAGNÓSTICO DO CLASSIFICADOR EEG")
    print("=" * 50)
    
    classifier = EEGClassifier()
    
    try:
        # Criar dataset
        print("📊 Criando dataset...")
        X, y, nomes = classifier.criar_dataset(limite=20)
        
        print(f"\n📈 Distribuição dos dados:")
        print(f"   Total de amostras: {len(y)}")
        print(f"   Classe 'Sim' (1): {np.sum(y == 1)}")
        print(f"   Classe 'Não' (0): {np.sum(y == 0)}")
        print(f"   Proporção Sim/Não: {np.sum(y == 1) / np.sum(y == 0):.2f}")
        
        # Criar e treinar modelo
        print("\n🧠 Treinando modelo...")
        classifier.criar_modelo(tipo_modelo='random_forest')
        resultados = classifier.treinar_modelo(X, y)
        
        # Testar predições em todos os sinais
        print("\n🔍 Testando predições em todos os sinais...")
        probabilidades_sim = []
        probabilidades_nao = []
        predicoes_corretas = []
        
        # Obter IDs reais dos sinais
        conexao = classifier.obter_conexao_db()
        cursor = conexao.cursor()
        
        ids_reais = []
        for categoria, label in [('S', 1), ('N', 0)]:
            cursor.execute("""
                SELECT s.id FROM sinais s
                JOIN usuarios u ON s.idusuario = u.id
                WHERE u.possui = %s
                ORDER BY s.id
                LIMIT 20
            """, (categoria,))
            ids_categoria = [row[0] for row in cursor.fetchall()]
            ids_reais.extend(ids_categoria)
        
        cursor.close()
        conexao.close()
        
        for i, (id_sinal, nome, label_real) in enumerate(zip(ids_reais, nomes, y)):
            resultado = classifier.prever_sinal(id_sinal)
            if resultado:
                prob = resultado['probabilidade']
                classe_pred = resultado['classe_predita']
                classe_real = 'Sim' if label_real == 1 else 'Não'
                
                # Coletar probabilidades por classe real
                if label_real == 1:  # Classe real é 'Sim'
                    probabilidades_sim.append(prob)
                else:  # Classe real é 'Não'
                    probabilidades_nao.append(1 - prob)  # Probabilidade da classe 'Não'
                
                # Verificar se predição está correta
                predicao_correta = (classe_pred == classe_real)
                predicoes_corretas.append(predicao_correta)
                
                print(f"   {nome}: Real={classe_real}, Pred={classe_pred}, "
                      f"Prob_Sim={prob:.3f}, Prob_Não={1-prob:.3f}, "
                      f"Correto={'✅' if predicao_correta else '❌'}")
        
        # Análise estatística das probabilidades
        print(f"\n📊 ANÁLISE DAS PROBABILIDADES:")
        
        if len(probabilidades_sim) > 0:
            print(f"   Sinais 'Sim' reais ({len(probabilidades_sim)}):")
            print(f"     Média prob. 'Sim': {np.mean(probabilidades_sim):.3f}")
            print(f"     Mediana prob. 'Sim': {np.median(probabilidades_sim):.3f}")
            print(f"     Min prob. 'Sim': {np.min(probabilidades_sim):.3f}")
            print(f"     Max prob. 'Sim': {np.max(probabilidades_sim):.3f}")
        else:
            print(f"   Sinais 'Sim' reais: Nenhum sinal válido encontrado")
        
        if len(probabilidades_nao) > 0:
            print(f"\n   Sinais 'Não' reais ({len(probabilidades_nao)}):")
            print(f"     Média prob. 'Não': {np.mean(probabilidades_nao):.3f}")
            print(f"     Mediana prob. 'Não': {np.median(probabilidades_nao):.3f}")
            print(f"     Min prob. 'Não': {np.min(probabilidades_nao):.3f}")
            print(f"     Max prob. 'Não': {np.max(probabilidades_nao):.3f}")
        else:
            print(f"\n   Sinais 'Não' reais: Nenhum sinal válido encontrado")
        
        if len(predicoes_corretas) > 0:
            print(f"\n   Acurácia geral: {np.mean(predicoes_corretas):.3f}")
        else:
            print(f"\n   Acurácia geral: Nenhuma predição válida")
        
        # Plotar distribuição das probabilidades apenas se houver dados
        if len(probabilidades_sim) > 0 or len(probabilidades_nao) > 0:
            plotar_distribuicao_probabilidades(probabilidades_sim, probabilidades_nao)
        else:
            print("   Não há dados suficientes para gerar gráficos")
        
        # Testar diferentes configurações de modelo
        testar_diferentes_modelos(X, y)
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

def plotar_distribuicao_probabilidades(probs_sim, probs_nao):
    """Plota a distribuição das probabilidades"""
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Histogramas
    plt.subplot(1, 2, 1)
    plt.hist(probs_sim, alpha=0.7, label='Sinais "Sim" reais', bins=10, color='green')
    plt.hist(probs_nao, alpha=0.7, label='Sinais "Não" reais', bins=10, color='red')
    plt.xlabel('Probabilidade')
    plt.ylabel('Frequência')
    plt.title('Distribuição das Probabilidades')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Box plots
    plt.subplot(1, 2, 2)
    data_to_plot = [probs_sim, probs_nao]
    labels = ['Sinais "Sim"', 'Sinais "Não"']
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel('Probabilidade')
    plt.title('Box Plot das Probabilidades')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/distribuicao_probabilidades.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("   Gráfico salvo em: static/distribuicao_probabilidades.png")

def testar_diferentes_modelos(X, y):
    """Testa diferentes configurações de modelo"""
    print(f"\n🧪 TESTANDO DIFERENTES CONFIGURAÇÕES DE MODELO:")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    
    modelos = {
        'Random Forest (padrão)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Random Forest (profundo)': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
        'Random Forest (conservador)': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        'MLP (pequeno)': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42),
        'MLP (grande)': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42)
    }
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for nome, modelo in modelos.items():
        scores = cross_val_score(modelo, X_scaled, y, cv=5, scoring='accuracy')
        print(f"   {nome}: {scores.mean():.3f} ± {scores.std():.3f}")
        
        # Treinar e testar um modelo específico
        if 'Random Forest (padrão)' in nome:
            modelo.fit(X_scaled, y)
            probas = modelo.predict_proba(X_scaled)
            probas_sim = probas[y == 1, 1]  # Probabilidades da classe 'Sim' para sinais 'Sim' reais
            probas_nao = probas[y == 0, 0]  # Probabilidades da classe 'Não' para sinais 'Não' reais
            
            print(f"     Prob. média 'Sim' para sinais 'Sim': {np.mean(probas_sim):.3f}")
            print(f"     Prob. média 'Não' para sinais 'Não': {np.mean(probas_nao):.3f}")

if __name__ == "__main__":
    testar_classificador() 