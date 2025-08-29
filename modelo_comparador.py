#!/usr/bin/env python3
"""
Comparador de modelos de rede neural para classificação EEG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ml_classifier import EEGClassifier
import os
from datetime import datetime

class ModeloComparador:
    """Classe para comparar diferentes modelos de ML/DL"""
    
    def __init__(self):
        self.modelos = {}
        self.resultados = {}
        self.historico_treinos = {}
        
    def adicionar_modelo(self, nome, tipo_modelo):
        """
        Adiciona um modelo para comparação
        
        Args:
            nome (str): Nome do modelo
            tipo_modelo (str): Tipo do modelo ('random_forest', 'mlp', 'cnn', 'lstm', 'hybrid')
        """
        print(f"➕ Adicionando modelo: {nome} ({tipo_modelo})")
        self.modelos[nome] = EEGClassifier()
        self.modelos[nome].criar_modelo(tipo_modelo)
        
    def treinar_todos_modelos(self, X, y, validation_split=0.2):
        """
        Treina todos os modelos com os mesmos dados
        
        Args:
            X (np.array): Features
            y (np.array): Labels
            validation_split (float): Proporção para validação
        """
        print(f"🚀 Treinando {len(self.modelos)} modelos...")
        
        for nome, modelo in self.modelos.items():
            print(f"\n{'='*50}")
            print(f"🧠 TREINANDO: {nome}")
            print(f"{'='*50}")
            
            try:
                resultado = modelo.treinar_modelo(X, y, validation_split)
                self.resultados[nome] = resultado
                
                # Salvar histórico se for modelo Keras
                if 'history' in resultado:
                    self.historico_treinos[nome] = resultado['history']
                    
                print(f"✅ {nome} treinado com sucesso!")
                
            except Exception as e:
                print(f"❌ Erro ao treinar {nome}: {e}")
                self.resultados[nome] = None
    
    def avaliar_modelos(self, X_teste, y_teste):
        """
        Avalia todos os modelos no conjunto de teste
        
        Args:
            X_teste (np.array): Features de teste
            y_teste (np.array): Labels de teste
        """
        print(f"\n🎯 Avaliando {len(self.modelos)} modelos no conjunto de teste...")
        
        metricas = {}
        
        for nome, modelo in self.modelos.items():
            if not modelo.is_trained:
                print(f"⚠️ {nome} não foi treinado, pulando...")
                continue
                
            try:
                # Fazer predições
                if hasattr(modelo.model, 'predict_proba'):
                    # Modelo scikit-learn
                    X_scaled = modelo.scaler.transform(X_teste)
                    y_pred = modelo.model.predict(X_scaled)
                    y_pred_proba = modelo.model.predict_proba(X_scaled)[:, 1]
                else:
                    # Modelo Keras
                    target_length = 100
                    X_reshaped = np.zeros((X_teste.shape[0], target_length, 1))
                    
                    for i in range(X_teste.shape[0]):
                        if X_teste.shape[1] >= target_length:
                            X_reshaped[i] = X_teste[i, :target_length].reshape(-1, 1)
                        else:
                            X_reshaped[i, :X_teste.shape[1], 0] = X_teste[i]
                    
                    y_pred_proba = modelo.model.predict(X_reshaped).flatten()
                    y_pred = (y_pred_proba > 0.5).astype(int)
                
                # Calcular métricas
                metricas[nome] = {
                    'accuracy': accuracy_score(y_teste, y_pred),
                    'precision': precision_score(y_teste, y_pred),
                    'recall': recall_score(y_teste, y_pred),
                    'f1': f1_score(y_teste, y_pred),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"✅ {nome} avaliado!")
                
            except Exception as e:
                print(f"❌ Erro ao avaliar {nome}: {e}")
                metricas[nome] = None
        
        return metricas
    
    def plotar_comparacao_metricas(self, metricas):
        """
        Plota comparação das métricas dos modelos
        """
        if not metricas:
            print("❌ Nenhuma métrica para plotar")
            return
        
        # Preparar dados
        modelos = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for nome, metrica in metricas.items():
            if metrica is not None:
                modelos.append(nome)
                accuracies.append(metrica['accuracy'])
                precisions.append(metrica['precision'])
                recalls.append(metrica['recall'])
                f1_scores.append(metrica['f1'])
        
        # Criar gráfico
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        ax1.bar(modelos, accuracies, color='skyblue')
        ax1.set_title('Acurácia')
        ax1.set_ylabel('Acurácia')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Precision
        ax2.bar(modelos, precisions, color='lightgreen')
        ax2.set_title('Precisão')
        ax2.set_ylabel('Precisão')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(precisions):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Recall
        ax3.bar(modelos, recalls, color='lightcoral')
        ax3.set_title('Recall')
        ax3.set_ylabel('Recall')
        ax3.set_ylim(0, 1)
        for i, v in enumerate(recalls):
            ax3.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # F1-Score
        ax4.bar(modelos, f1_scores, color='gold')
        ax4.set_title('F1-Score')
        ax4.set_ylabel('F1-Score')
        ax4.set_ylim(0, 1)
        for i, v in enumerate(f1_scores):
            ax4.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        # Salvar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho = f"static/comparacao_modelos_{timestamp}.png"
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Gráfico salvo em: {caminho}")
        return caminho
    
    def plotar_curvas_treino(self):
        """
        Plota curvas de treinamento para modelos Keras
        """
        if not self.historico_treinos:
            print("❌ Nenhum histórico de treinamento disponível")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        for nome, history in self.historico_treinos.items():
            # Loss
            ax1.plot(history.history['loss'], label=f'{nome} (train)')
            if 'val_loss' in history.history:
                ax1.plot(history.history['val_loss'], label=f'{nome} (val)')
            ax1.set_title('Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy
            if 'accuracy' in history.history:
                ax2.plot(history.history['accuracy'], label=f'{nome} (train)')
                if 'val_accuracy' in history.history:
                    ax2.plot(history.history['val_accuracy'], label=f'{nome} (val)')
                ax2.set_title('Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                ax2.grid(True)
        
        plt.tight_layout()
        
        # Salvar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        caminho = f"static/curvas_treino_{timestamp}.png"
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Curvas de treinamento salvas em: {caminho}")
        return caminho
    
    def gerar_relatorio(self, metricas):
        """
        Gera relatório comparativo dos modelos
        """
        print("\n" + "="*80)
        print("📊 RELATÓRIO COMPARATIVO DOS MODELOS")
        print("="*80)
        
        # Tabela de métricas
        dados = []
        for nome, metrica in metricas.items():
            if metrica is not None:
                dados.append({
                    'Modelo': nome,
                    'Acurácia': f"{metrica['accuracy']:.4f}",
                    'Precisão': f"{metrica['precision']:.4f}",
                    'Recall': f"{metrica['recall']:.4f}",
                    'F1-Score': f"{metrica['f1']:.4f}"
                })
        
        if dados:
            df = pd.DataFrame(dados)
            print(df.to_string(index=False))
            
            # Melhor modelo por métrica
            print("\n🏆 MELHORES MODELOS POR MÉTRICA:")
            for metrica in ['accuracy', 'precision', 'recall', 'f1']:
                melhor = max(metricas.items(), key=lambda x: x[1][metrica] if x[1] else 0)
                print(f"   {metrica.upper()}: {melhor[0]} ({melhor[1][metrica]:.4f})")
        else:
            print("❌ Nenhum modelo foi avaliado com sucesso")
    
    def salvar_modelos(self, diretorio="modelos_salvos"):
        """
        Salva todos os modelos treinados
        """
        if not os.path.exists(diretorio):
            os.makedirs(diretorio)
        
        for nome, modelo in self.modelos.items():
            if modelo.is_trained:
                caminho = os.path.join(diretorio, f"{nome}_modelo.pkl")
                modelo.salvar_modelo(caminho)
                print(f"💾 Modelo {nome} salvo em: {caminho}")

def main():
    """Função principal para demonstrar o comparador"""
    print("🧠 COMPARADOR DE MODELOS DE REDE NEURAL")
    print("="*60)
    
    # Criar comparador
    comparador = ModeloComparador()
    
    # Adicionar modelos
    comparador.adicionar_modelo("Random Forest", "random_forest")
    comparador.adicionar_modelo("MLP", "mlp")
    comparador.adicionar_modelo("CNN", "cnn")
    comparador.adicionar_modelo("LSTM", "lstm")
    comparador.adicionar_modelo("Hybrid CNN-LSTM", "hybrid")
    
    # Criar dataset de exemplo
    print("\n📊 Criando dataset de exemplo...")
    from ml_classifier import EEGClassifier
    classifier_temp = EEGClassifier()
    X, y, _ = classifier_temp.criar_dataset(limite=50)
    
    if X is not None and len(X) > 0:
        # Treinar todos os modelos
        comparador.treinar_todos_modelos(X, y)
        
        # Avaliar modelos
        metricas = comparador.avaliar_modelos(X, y)
        
        # Gerar visualizações
        comparador.plotar_comparacao_metricas(metricas)
        comparador.plotar_curvas_treino()
        
        # Gerar relatório
        comparador.gerar_relatorio(metricas)
        
        # Salvar modelos
        comparador.salvar_modelos()
        
    else:
        print("❌ Não foi possível criar dataset. Verifique se há dados no banco.")

if __name__ == "__main__":
    main()
