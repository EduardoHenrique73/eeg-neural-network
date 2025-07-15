#!/usr/bin/env python3
"""
Sistema de Machine Learning para Classificação de Sinais EEG
Usa entropia de Shannon e outras features para distinguir entre grupos
Versão simplificada usando scikit-learn (sem TensorFlow)
"""

import numpy as np
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from dinamica_simbolica import aplicar_dinamica_simbolica
import os
import pickle

class EEGClassifier:
    """Classificador de sinais EEG usando machine learning"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def obter_conexao_db(self):
        """Conecta ao banco de dados PostgreSQL"""
        return psycopg2.connect(
            dbname="eeg-projeto",
            user="postgres",
            password="EEG@321",
            host="localhost",
            port="5432"
        )
    
    def extrair_features_sinal(self, id_sinal):
        """
        Extrai features de um sinal EEG específico
        
        Args:
            id_sinal (int): ID do sinal no banco
            
        Returns:
            dict: Dicionário com as features extraídas
        """
        try:
            # Aplica dinâmica simbólica
            resultado = aplicar_dinamica_simbolica(id_sinal, m=3)
            
            # Verifica se o resultado é válido
            if not resultado or len(resultado['sequencia_binaria']) == 0:
                print(f"Sinal {id_sinal}: sequência binária vazia")
                return None
            
            # Features básicas
            features = {
                'entropia_shannon': resultado['entropia'],
                'limiar': resultado['limiar'],
                'total_amostras': len(resultado['sequencia_binaria']),
                'total_padroes': len(resultado['palavras_decimais']),
                'padroes_unicos': len(resultado['frequencias'])
            }
            
            # Estatísticas dos valores brutos (precisamos buscar novamente)
            conexao = self.obter_conexao_db()
            cursor = conexao.cursor()
            cursor.execute("SELECT valor FROM valores_sinais WHERE idsinal = %s", (id_sinal,))
            valores_brutos = np.array([row[0] for row in cursor.fetchall()])
            cursor.close()
            conexao.close()
            
            # Verifica se há dados válidos
            if len(valores_brutos) == 0:
                print(f"Sinal {id_sinal}: valores brutos vazios")
                return None
            
            # Features estatísticas dos valores brutos
            features.update({
                'media_valores': np.mean(valores_brutos),
                'desvio_padrao': np.std(valores_brutos),
                'variancia': np.var(valores_brutos),
                'skewness': self._calcular_skewness(valores_brutos),
                'kurtosis': self._calcular_kurtosis(valores_brutos),
                'amplitude': np.max(valores_brutos) - np.min(valores_brutos),
                'rms': np.sqrt(np.mean(valores_brutos**2))
            })
            
            # Features da sequência binária
            sequencia_binaria = np.array([int(b) for b in resultado['sequencia_binaria']])
            features.update({
                'proporcao_uns': np.mean(sequencia_binaria),
                'transicoes': self._contar_transicoes(sequencia_binaria),
                'comprimento_sequencia': len(sequencia_binaria)
            })
            
            # Features das frequências dos padrões
            freq_values = list(resultado['frequencias'].values())
            if len(freq_values) == 0:
                print(f"Sinal {id_sinal}: frequências vazias")
                return None
                
            features.update({
                'max_frequencia': np.max(freq_values),
                'min_frequencia': np.min(freq_values),
                'std_frequencias': np.std(freq_values),
                'entropia_frequencias': self._calcular_entropia_frequencias(freq_values)
            })
            
            return features
            
        except Exception as e:
            print(f"Erro ao extrair features do sinal {id_sinal}: {e}")
            return None
    
    def _calcular_skewness(self, dados):
        """Calcula a assimetria dos dados"""
        media = np.mean(dados)
        desvio = np.std(dados)
        if desvio == 0:
            return 0
        return np.mean(((dados - media) / desvio) ** 3)
    
    def _calcular_kurtosis(self, dados):
        """Calcula a curtose dos dados"""
        media = np.mean(dados)
        desvio = np.std(dados)
        if desvio == 0:
            return 0
        return np.mean(((dados - media) / desvio) ** 4) - 3
    
    def _contar_transicoes(self, sequencia):
        """Conta o número de transições 0->1 e 1->0"""
        if len(sequencia) < 2:
            return 0
        return np.sum(np.abs(np.diff(sequencia)))
    
    def _calcular_entropia_frequencias(self, frequencias):
        """Calcula entropia das frequências (redundância)"""
        if not frequencias:
            return 0
        freq_array = np.array(frequencias)
        freq_array = freq_array[freq_array > 0]
        if len(freq_array) == 0:
            return 0
        freq_array = freq_array / np.sum(freq_array)
        return -np.sum(freq_array * np.log2(freq_array))
    
    def criar_dataset(self, limite=None):
        """
        Cria dataset de treinamento com todos os sinais
        
        Args:
            limite (int, optional): Limite de sinais por categoria
            
        Returns:
            tuple: (X, y) - features e labels
        """
        print("📊 Criando dataset de treinamento...")
        
        conexao = self.obter_conexao_db()
        cursor = conexao.cursor()
        
        # Busca sinais de ambas as categorias separadamente
        features_list = []
        labels = []
        nomes_sinais = []
        
        for categoria, label in [('S', 1), ('N', 0)]:
            query = """
                SELECT s.id, s.nome, u.possui
                FROM sinais s
                JOIN usuarios u ON s.idusuario = u.id
                WHERE u.possui = %s
                ORDER BY s.id
            """
            
            if limite:
                query += f" LIMIT {limite}"
            
            cursor.execute(query, (categoria,))
            sinais = cursor.fetchall()
            
            print(f"   Categoria '{categoria}': {len(sinais)} sinais")
            
            for id_sinal, nome, possui in sinais:
                print(f"     Processando: {nome} (ID: {id_sinal})")
                
                features = self.extrair_features_sinal(id_sinal)
                if features:
                    features_list.append(features)
                    labels.append(label)
                    nomes_sinais.append(nome)
                else:
                    print(f"       ⚠️  Falhou ao extrair features")
        
        cursor.close()
        conexao.close()
        
        if not features_list:
            raise ValueError("Nenhuma feature foi extraída com sucesso!")
        
        # Converte para DataFrame
        df = pd.DataFrame(features_list)
        self.feature_names = df.columns.tolist()
        
        X = df.values
        y = np.array(labels)
        
        print(f"✅ Dataset criado: {X.shape[0]} amostras, {X.shape[1]} features")
        print(f"   Distribuição: {np.sum(y == 1)} 'Sim', {np.sum(y == 0)} 'Não'")
        
        return X, y, nomes_sinais
    
    def criar_modelo(self, tipo_modelo='random_forest'):
        """
        Cria o modelo de classificação
        
        Args:
            tipo_modelo (str): 'random_forest' ou 'neural_network'
        """
        print(f"🧠 Criando modelo: {tipo_modelo}")
        
        if tipo_modelo == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif tipo_modelo == 'neural_network':
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            )
        else:
            raise ValueError("Tipo de modelo deve ser 'random_forest' ou 'neural_network'")
        
        print("✅ Modelo criado com sucesso!")
    
    def treinar_modelo(self, X, y, validation_split=0.2):
        """
        Treina o modelo de classificação
        
        Args:
            X (np.array): Features de treinamento
            y (np.array): Labels de treinamento
            validation_split (float): Proporção para validação
        """
        print("🚀 Iniciando treinamento do modelo...")
        
        # Divisão treino/validação
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Normalização
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Treinamento
        self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Avaliação
        y_pred = self.model.predict(X_val_scaled)
        
        print("\n📊 RESULTADOS DO TREINAMENTO:")
        print(f"   Acurácia: {accuracy_score(y_val, y_pred):.4f}")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_pred': y_pred
        }
    
    def avaliar_modelo(self, X, y):
        """
        Avalia o modelo treinado
        
        Args:
            X (np.array): Features de teste
            y (np.array): Labels de teste
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]  # Probabilidade da classe positiva
        
        print("\n🎯 AVALIAÇÃO DO MODELO:")
        print(f"Acurácia: {accuracy_score(y, y_pred):.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y, y_pred, target_names=['Não', 'Sim']))
        
        # Matriz de confusão
        cm = confusion_matrix(y, y_pred)
        self._plotar_matriz_confusao(cm)
        
        return y_pred, y_pred_proba
    
    def _plotar_matriz_confusao(self, cm):
        """Plota matriz de confusão"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Não', 'Sim'],
                   yticklabels=['Não', 'Sim'])
        plt.title('Matriz de Confusão - Classificador EEG')
        plt.ylabel('Valor Real')
        plt.xlabel('Valor Predito')
        
        # Salvar
        caminho = os.path.join("static", "matriz_confusao.png")
        plt.savefig(caminho, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"   Matriz de confusão salva em: {caminho}")
    
    def prever_sinal(self, id_sinal):
        """
        Faz predição para um sinal específico
        
        Args:
            id_sinal (int): ID do sinal
            
        Returns:
            dict: Resultado da predição
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        features = self.extrair_features_sinal(id_sinal)
        if not features:
            return None
        
        # Converte para array
        X = np.array([list(features.values())])
        X_scaled = self.scaler.transform(X)
        
        # Predição
        classe_predita = self.model.predict(X_scaled)[0]
        probabilidade = self.model.predict_proba(X_scaled)[0][1]  # Probabilidade da classe 'Sim'
        
        return {
            'classe_predita': 'Sim' if classe_predita == 1 else 'Não',
            'probabilidade': probabilidade,
            'features': features
        }
    
    def salvar_modelo(self, caminho='modelo_eeg.pkl'):
        """Salva o modelo treinado"""
        if self.is_trained:
            with open(caminho, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'is_trained': self.is_trained
                }, f)
            print(f"✅ Modelo salvo em: {caminho}")
        else:
            print("❌ Modelo não foi treinado ainda!")
    
    def carregar_modelo(self, caminho='modelo_eeg.pkl'):
        """Carrega um modelo salvo"""
        if os.path.exists(caminho):
            with open(caminho, 'rb') as f:
                dados = pickle.load(f)
            self.model = dados['model']
            self.scaler = dados['scaler']
            self.feature_names = dados['feature_names']
            self.is_trained = dados['is_trained']
            print(f"✅ Modelo carregado de: {caminho}")
        else:
            print(f"❌ Arquivo não encontrado: {caminho}")

def main():
    """Função principal para treinar e testar o classificador"""
    print("🧠 SISTEMA DE CLASSIFICAÇÃO EEG")
    print("=" * 50)
    
    # Criar classificador
    classifier = EEGClassifier()
    
    try:
        # Criar dataset
        X, y, nomes = classifier.criar_dataset(limite=20)  # 20 sinais por categoria
        
        # Criar modelo (Random Forest é mais estável)
        classifier.criar_modelo(tipo_modelo='random_forest')
        
        # Treinar modelo
        resultados = classifier.treinar_modelo(X, y)
        
        # Avaliar modelo
        classifier.avaliar_modelo(X, y)
        
        # Salvar modelo
        classifier.salvar_modelo()
        
        # Teste com alguns sinais
        print("\n🔍 TESTE DE PREDIÇÃO:")
        for i in range(min(5, len(nomes))):
            resultado = classifier.prever_sinal(i + 1)  # IDs começam em 1
            if resultado:
                print(f"   {nomes[i]}: {resultado['classe_predita']} "
                      f"(prob: {resultado['probabilidade']:.3f})")
        
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main() 