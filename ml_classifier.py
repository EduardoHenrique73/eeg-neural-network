


import numpy as np
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from dinamica_simbolica import aplicar_dinamica_simbolica
import os
import pickle
from config import config
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, BatchNormalization, GlobalAveragePooling1D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class EEGClassifier:
    """Classificador de sinais EEG usando machine learning"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.tipo_modelo_keras = None  # Para criar modelo dinamicamente
        
    def obter_conexao_db(self):
        """Conecta ao banco de dados PostgreSQL"""
        return psycopg2.connect(**config.get_db_connection_string())
    
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
                'entropia_frequencias': self._calcular_entropia_shannon(freq_values)
            })
            
            return features
            
        except Exception as e:
            print(f"Erro ao extrair features do sinal {id_sinal}: {str(e)}")
            return None
    
    def _calcular_skewness(self, data):
        """Calcula o skewness (assimetria) dos dados"""
        n = len(data)
        if n < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        skewness = (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
        return skewness
    
    def _calcular_kurtosis(self, data):
        """Calcula o kurtosis (curtose) dos dados"""
        n = len(data)
        if n < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        kurtosis = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std) ** 4) - (3 * (n-1)**2 / ((n-2) * (n-3)))
        return kurtosis
    
    def _contar_transicoes(self, sequencia):
        """Conta o número de transições 0->1 e 1->0 na sequência binária"""
        if len(sequencia) < 2:
            return 0
        
        transicoes = 0
        for i in range(1, len(sequencia)):
            if sequencia[i] != sequencia[i-1]:
                transicoes += 1
        
        return transicoes
    
    def _calcular_entropia_shannon(self, valores):
        """Calcula a entropia de Shannon de uma lista de valores"""
        if len(valores) == 0:
            return 0.0
        
        # Normaliza os valores para probabilidades
        total = sum(valores)
        if total == 0:
            return 0.0
        
        probabilidades = [v/total for v in valores]
        
        # Calcula entropia
        entropia = 0.0
        for p in probabilidades:
            if p > 0:
                entropia -= p * np.log2(p)
        
        return entropia
    
    def criar_dataset(self, limite=10):
        """
        Cria dataset de treinamento balanceado a partir dos dados do banco
        
        Args:
            limite (int): Número máximo de sinais por categoria
            
        Returns:
            tuple: (X, y, feature_names)
        """
        try:
            conexao = self.obter_conexao_db()
            cursor = conexao.cursor()
            
            # Busca sinais balanceados por categoria
            sinais = []
            for categoria in ['S', 'N']:
                cursor.execute("""
                    SELECT s.id, s.nome, u.possui
                    FROM sinais s
                    JOIN usuarios u ON s.idusuario = u.id
                    WHERE u.possui = %s
                    LIMIT %s
                """, (categoria, limite))
                
                sinais_categoria = cursor.fetchall()
                sinais.extend(sinais_categoria)
            
            cursor.close()
            conexao.close()
            
            if not sinais:
                print("Nenhum sinal encontrado no banco de dados")
                return None, None, None
            
            features_list = []
            labels = []
            
            for id_sinal, nome, possui in sinais:
                print(f"Processando sinal {id_sinal}: {nome}")
                
                features = self.extrair_features_sinal(id_sinal)
                if features is None:
                    continue
                
                features_list.append(features)
                labels.append(1 if possui == 'S' else 0)
            
            if not features_list:
                print("Nenhuma feature válida extraída")
                return None, None, None
            
            # Converte para arrays
            X = np.array([list(features.values()) for features in features_list])
            y = np.array(labels)
            
            # Guarda nomes das features
            self.feature_names = list(features_list[0].keys())
            
            print(f"Dataset criado: {X.shape[0]} amostras, {X.shape[1]} features")
            print(f"Distribuição das classes: {np.bincount(y)}")
            
            return X, y, self.feature_names
            
        except Exception as e:
            print(f"Erro ao criar dataset: {str(e)}")
            return None, None, None
    
    def criar_modelo(self, tipo_modelo='random_forest'):
        """
        Cria o modelo de classificação
        
        Args:
            tipo_modelo (str): 'random_forest', 'mlp', 'cnn', 'lstm', 'hybrid', ou 'mlp_tabular'
        """
        print(f"🧠 Criando modelo: {tipo_modelo}")
        
        if tipo_modelo == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif tipo_modelo == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            )
        elif tipo_modelo in ['cnn', 'lstm', 'hybrid', 'mlp_tabular']:
            # Para modelos TensorFlow/Keras - criaremos dinamicamente no treino
            self.tipo_modelo_keras = tipo_modelo
            self.model = None  # Será criado no treino com input shape correto
        else:
            raise ValueError("Tipo de modelo deve ser 'random_forest', 'mlp', 'cnn', 'lstm', 'hybrid', ou 'mlp_tabular'")
        
        print("✅ Modelo criado com sucesso!")
    
    def _criar_modelo_keras(self, tipo_modelo, n_features):
        """
        Cria modelos usando TensorFlow/Keras com input shape dinâmico
        """
        if tipo_modelo == 'cnn':
            return self._criar_cnn(n_features)
        elif tipo_modelo == 'lstm':
            return self._criar_lstm(n_features)
        elif tipo_modelo == 'hybrid':
            return self._criar_hybrid(n_features)
        elif tipo_modelo == 'mlp_tabular':
            return self._criar_mlp_tabular(n_features)
    
    def _criar_mlp_tabular(self, n_features):
        """
        Cria MLP otimizado para dados tabulares (melhor que CNN para features estatísticas)
        """
        model = Sequential([
            Input(shape=(n_features,)),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def _criar_cnn(self, n_features):
        """
        Cria uma CNN 1D otimizada para dados tabulares (sem padding desnecessário)
        """
        model = Sequential([
            Input(shape=(n_features, 1)),
            # Para tabular, CNN pequena funciona melhor
            Conv1D(32, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),
            
            Conv1D(64, 3, activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dropout(0.4),
            
            Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def _criar_lstm(self, n_features):
        """
        Cria uma rede LSTM otimizada para dados tabulares
        """
        model = Sequential([
            Input(shape=(n_features, 1)),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def _criar_hybrid(self, n_features):
        """
        Cria uma rede híbrida CNN-LSTM para dados tabulares
        """
        model = Sequential([
            Input(shape=(n_features, 1)),
            # Camada CNN para extração de features
            Conv1D(32, 3, activation='relu', padding='same'),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu', padding='same'),
            MaxPooling1D(2),
            
            # Camada LSTM para processamento temporal
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Camadas densas para classificação
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def treinar_modelo(self, X, y, validation_split=0.2):
        """
        Treina o modelo com os dados fornecidos
        
        Args:
            X (np.array): Features de entrada
            y (np.array): Labels de saída
            validation_split (float): Proporção para validação
        """
        if X is None or y is None or len(X) == 0:
            print("❌ Dados de treinamento inválidos")
            return
        
        print(f"🚀 Iniciando treinamento com {len(X)} amostras...")
        
        # Verifica se é modelo sklearn ou keras
        if self.tipo_modelo_keras is not None:  # Modelo Keras
            self._treinar_keras(X, y, validation_split)
        else:  # Modelo sklearn
            self._treinar_sklearn(X, y, validation_split)
    
    def _treinar_sklearn(self, X, y, validation_split=0.2):
        """
        Treina modelos sklearn (Random Forest, MLP)
        """
        try:
            # Verifica se há pelo menos 2 classes
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                print(f"❌ Dataset tem apenas uma classe: {unique_classes}")
                self.is_trained = False
                return
            
            # Escala os dados
            X_scaled = self.scaler.fit_transform(X)
            
            # Split treino/validação (sem stratify se apenas uma classe)
            if len(unique_classes) >= 2:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=validation_split, random_state=42, stratify=y
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y, test_size=validation_split, random_state=42
                )
            
            print(f"📊 Treinando com {len(X_train)} amostras, validando com {len(X_val)}")
            
            # Treina o modelo
            self.model.fit(X_train, y_train)
            
            # Avalia
            y_pred = self.model.predict(X_val)
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            # Métricas
            acc = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            print("\n📊 RESULTADOS DO TREINAMENTO (SKLEARN):")
            print(f"   Acurácia: {acc:.4f}")
            print(f"   Precisão: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            
            self.is_trained = True
            print("✅ Modelo sklearn treinado com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro no treinamento sklearn: {str(e)}")
            self.is_trained = False
    
    def _treinar_keras(self, X, y, validation_split=0.2):
        """
        Treina modelos Keras (CNN, LSTM, MLP Tabular)
        """
        try:
            # Verifica se há pelo menos 2 classes
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                print(f"❌ Dataset tem apenas uma classe: {unique_classes}")
                self.is_trained = False
                return
            
            y = y.astype('float32')
            n_samples, n_features = X.shape
            
            print(f"📊 Treinando modelo Keras: {n_samples} amostras, {n_features} features")
            
            # Escala os dados
            X_scaled = self.scaler.fit_transform(X)
            
            # Reshape baseado no tipo de modelo
            if self.tipo_modelo_keras == 'mlp_tabular':
                # MLP tabular usa features diretamente
                X_reshaped = X_scaled
            else:
                # CNN/LSTM/Hybrid usa (samples, timesteps, features)
                X_reshaped = X_scaled.reshape(n_samples, n_features, 1)
            
            # Cria modelo com input shape correto
            if self.model is None:
                self.model = self._criar_modelo_keras(self.tipo_modelo_keras, n_features)
                print(f"✅ Modelo {self.tipo_modelo_keras} criado com input shape: {X_reshaped.shape}")
            
            # Split treino/validação (sem stratify se apenas uma classe)
            if len(unique_classes) >= 2:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_reshaped, y, test_size=validation_split, random_state=42, stratify=y
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_reshaped, y, test_size=validation_split, random_state=42
                )
            
            # Callbacks para otimização
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=15, 
                restore_best_weights=True, 
                min_delta=0.001
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-7, 
                verbose=1
            )
            
            # Treina o modelo
            history = self.model.fit(
                X_train, y_train,
                epochs=200,
                batch_size=8,  # Batch menor para dataset pequeno
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            self.is_trained = True
            
            # Avalia o modelo
            y_pred_proba = self.model.predict(X_val).ravel()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            acc = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            print("\n📊 RESULTADOS DO TREINAMENTO (KERAS):")
            print(f"   Acurácia: {acc:.4f}")
            print(f"   Precisão: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   Loss final: {history.history['loss'][-1]:.4f}")
            print(f"   Val Loss final: {history.history['val_loss'][-1]:.4f}")
            
            print(f"✅ Modelo {self.tipo_modelo_keras} treinado com sucesso!")
            
        except Exception as e:
            print(f"❌ Erro no treinamento Keras: {str(e)}")
            self.is_trained = False
    
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
        
        caminho = os.path.join("static", "matriz_confusao.png")
        plt.savefig(caminho, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"   Matriz de confusão salva em: {caminho}")
    
    def prever_sinal(self, id_sinal):
        """
        Faz predição para um sinal específico
        """
        try:
            # Extrair features do sinal
            features = self.extrair_features_sinal(id_sinal)
            if features is None:
                return None
            
            return self.prever_com_features(features)
            
        except Exception as e:
            print(f"❌ Erro na predição do sinal {id_sinal}: {e}")
            return None

    def prever_com_features(self, features):
        """
        Faz predição usando features já extraídas
        """
        try:
            if not self.is_trained:
                print("⚠️ Modelo não está treinado")
                return None
            
            # Preparar dados para predição
            x = np.array([list(features.values())])
            
            if self.tipo_modelo_keras is not None:
                # Modelo Keras
                x_scaled = self.scaler.transform(x)
                
                if self.tipo_modelo_keras == 'mlp_tabular':
                    # MLP Tabular: usar shape (n_samples, n_features)
                    X_reshaped = x_scaled
                else:
                    # CNN/LSTM/Hybrid: usar shape (n_samples, n_features, 1)
                    X_reshaped = x_scaled.reshape(1, x_scaled.shape[1], 1)
                
                probabilidade = float(self.model.predict(X_reshaped)[0, 0])
            else:
                # Modelo sklearn
                x_scaled = self.scaler.transform(x)
                probabilidade = float(self.model.predict_proba(x_scaled)[0, 1])
            
            # Determinar classe
            classe_predita = 'Sim' if probabilidade >= 0.5 else 'Não'
            
            return {
                'classe_predita': classe_predita,
                'probabilidade': probabilidade,
                'features': features
            }
            
        except Exception as e:
            print(f"❌ Erro na predição com features: {e}")
            return None
    
    def salvar_modelo(self, caminho_arquivo):
        """
        Salva o modelo treinado
        
        Args:
            caminho_arquivo (str): Caminho para salvar o modelo
        """
        try:
            if not self.is_trained:
                print("❌ Modelo não está treinado")
                return False
            
            # Salva informações do modelo
            modelo_info = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'tipo_modelo_keras': self.tipo_modelo_keras
            }
            
            with open(caminho_arquivo, 'wb') as f:
                pickle.dump(modelo_info, f)
            
            print(f"✅ Modelo salvo em: {caminho_arquivo}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar modelo: {str(e)}")
            return False
    
    def carregar_modelo(self, caminho_arquivo):
        """
        Carrega um modelo salvo
        
        Args:
            caminho_arquivo (str): Caminho do arquivo do modelo
            
        Returns:
            bool: True se carregou com sucesso
        """
        try:
            if not os.path.exists(caminho_arquivo):
                print(f"❌ Arquivo não encontrado: {caminho_arquivo}")
                return False
            
            with open(caminho_arquivo, 'rb') as f:
                modelo_info = pickle.load(f)
            
            self.model = modelo_info['model']
            self.scaler = modelo_info['scaler']
            self.feature_names = modelo_info['feature_names']
            self.is_trained = modelo_info['is_trained']
            self.tipo_modelo_keras = modelo_info.get('tipo_modelo_keras')
            
            print(f"✅ Modelo carregado de: {caminho_arquivo}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {str(e)}")
            return False

def main():
    """Função principal para treinar e testar o classificador"""
    print("🧠 SISTEMA DE CLASSIFICAÇÃO EEG")
    print("=" * 50)
    
    classifier = EEGClassifier()
    
    try:
        # Criar dataset
        X, y, nomes = classifier.criar_dataset(limite=20)  
        
        classifier.criar_modelo(tipo_modelo='random_forest')
        
        resultados = classifier.treinar_modelo(X, y)
        
        classifier.avaliar_modelo(X, y)
        
        classifier.salvar_modelo('modelo_eeg.pkl')
        
     
        print("\n🔍 TESTE DE PREDIÇÃO:")
        for i in range(min(5, len(nomes))):
            resultado = classifier.prever_sinal(i + 1) 
            if resultado:
                print(f"   {nomes[i]}: {resultado['classe_predita']} "
                      f"(prob: {resultado['probabilidade']:.3f})")
        
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main() 