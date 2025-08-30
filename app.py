from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import psycopg2
import threading
import time
from datetime import datetime
from dinamica_simbolica import aplicar_dinamica_simbolica
from modulo_funcoes import gerar_grafico_interativo
from ml_classifier import EEGClassifier
from testes_sistema import TestadorSistema
from modelo_comparador import ModeloComparador
import numpy as np
import uuid
from config import config

app = Flask(__name__)

# Configuração para upload de arquivos
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Instância global do classificador
classifier = EEGClassifier()

# Variáveis globais para logs e status
retraining_logs = []
retraining_status = "idle"  # idle, running, completed, error
test_logs = []
test_status = "idle"

# Modelos globais para evitar retreinamento desnecessário
classifier_cnn = None
classifier_lstm = None

def obter_conexao_db():
    """Conecta ao banco de dados PostgreSQL."""
    return psycopg2.connect(**config.get_db_connection_string())

def inicializar_classificador():
    """Inicializa o classificador, carregando modelo salvo ou treinando novo."""
    global retraining_logs, retraining_status
    
    def log_retraining(mensagem):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {mensagem}"
        retraining_logs.append(log_entry)
        print(log_entry)
    
    try:
        log_retraining("🔍 Tentando carregar modelo existente...")
        classifier.carregar_modelo(config.MODEL_PATH)
        if classifier.is_trained:
            log_retraining("✅ Modelo carregado com sucesso!")
            return True
        else:
            log_retraining("⚠️ Modelo não estava treinado, iniciando treinamento...")
    except Exception as e:
        log_retraining(f"⚠️ Erro ao carregar modelo: {e}")
    
    try:
        log_retraining("📊 Criando dataset de treinamento...")
        X, y, _ = classifier.criar_dataset(limite=20)
        log_retraining(f"✅ Dataset criado: {X.shape[0]} amostras, {X.shape[1]} features")
        
        log_retraining("🧠 Criando modelo Random Forest...")
        classifier.criar_modelo(tipo_modelo='random_forest')
        
        log_retraining("🚀 Iniciando treinamento...")
        resultado_treino = classifier.treinar_modelo(X, y)
        log_retraining("✅ Treinamento concluído com sucesso!")
        
        log_retraining("💾 Salvando modelo...")
        classifier.salvar_modelo(config.MODEL_PATH)
        log_retraining("✅ Modelo salvo com sucesso!")
        
        return True
    except Exception as e:
        log_retraining(f"❌ Erro durante treinamento: {e}")
        return False

@app.route("/")
def home():
    categoria = request.args.get("categoria")
    try:
        resultado = gerar_grafico_interativo(limite=10, filtro_categoria=categoria)
        graficos_html = resultado['graficos_html']
        histogramas = []
        
        # Criar modelos adicionais para comparação (APENAS UMA VEZ, fora do loop)
        global classifier_cnn, classifier_lstm
        
        # Inicializar MLP Tabular se necessário (melhor para dados tabulares)
        if classifier_cnn is None:
                   try:
                       print("🧠 Criando modelo MLP Tabular...")
                       classifier_cnn = EEGClassifier()
                       classifier_cnn.criar_modelo('mlp_tabular')
                       
                       # Tentar carregar MLP salvo
                       if classifier_cnn.carregar_modelo('modelo_mlp_tabular.pkl'):
                           print("✅ MLP Tabular carregado do arquivo salvo!")
                       else:
                           # Treinar novo MLP
                           X_temp, y_temp, _ = classifier_cnn.criar_dataset(limite=20)
                           if X_temp is not None and len(X_temp) > 0:
                               print(f"📊 Treinando MLP Tabular com {X_temp.shape[0]} amostras...")
                               classifier_cnn.treinar_modelo(X_temp, y_temp)
                               classifier_cnn.salvar_modelo('modelo_mlp_tabular.pkl')
                               print("✅ MLP Tabular treinado e salvo!")
                           else:
                               print("❌ Não foi possível criar dataset para MLP")
                               classifier_cnn = None
                   except Exception as e:
                       print(f"❌ Erro ao criar MLP Tabular: {e}")
                       classifier_cnn = None
        
        # Inicializar LSTM se necessário
        if classifier_lstm is None:
            try:
                print("🧠 Criando modelo LSTM...")
                classifier_lstm = EEGClassifier()
                classifier_lstm.criar_modelo('lstm')
                
                # Tentar carregar LSTM salvo
                if classifier_lstm.carregar_modelo('modelo_lstm.pkl'):
                    print("✅ LSTM carregado do arquivo salvo!")
                else:
                    # Treinar novo LSTM
                    X_temp, y_temp, _ = classifier_lstm.criar_dataset(limite=20)
                    if X_temp is not None and len(X_temp) > 0:
                        print(f"📊 Treinando LSTM com {X_temp.shape[0]} amostras...")
                        classifier_lstm.treinar_modelo(X_temp, y_temp)
                        classifier_lstm.salvar_modelo('modelo_lstm.pkl')
                        print("✅ LSTM treinado e salvo!")
                    else:
                        print("❌ Não foi possível criar dataset para LSTM")
                        classifier_lstm = None
            except Exception as e:
                print(f"❌ Erro ao criar LSTM: {e}")
                classifier_lstm = None
        
        # Agora processar cada sinal (sem criar modelos novamente)
        for sinal in resultado['dados_sinais']:
            try:
                resultado_ds = aplicar_dinamica_simbolica(sinal['id'], m=3)
                if resultado_ds is None:
                    continue
                predicao = None
                predicao_cnn = None
                predicao_lstm = None
                
                if classifier.is_trained:
                    try:
                        predicao = classifier.prever_sinal(sinal['id'])
                    except Exception:
                        pass
                
                # Fazer predições com modelos treinados
                try:
                    if classifier_cnn and classifier_cnn.is_trained:
                        print(f"🔮 Fazendo predição MLP Tabular para sinal {sinal['id']}...")
                        predicao_cnn = classifier_cnn.prever_sinal(sinal['id'])
                        print(f"✅ Predição MLP Tabular: {predicao_cnn}")
                    else:
                        print("❌ MLP Tabular não está treinado")
                except Exception as e:
                    print(f"❌ Erro na predição MLP Tabular: {e}")
                
                try:
                    if classifier_lstm and classifier_lstm.is_trained:
                        print(f"🔮 Fazendo predição LSTM para sinal {sinal['id']}...")
                        predicao_lstm = classifier_lstm.prever_sinal(sinal['id'])
                        print(f"✅ Predição LSTM: {predicao_lstm}")
                    else:
                        print("❌ LSTM não está treinado")
                except Exception as e:
                    print(f"❌ Erro na predição LSTM: {e}")
                sequencia_binaria = resultado_ds.get('sequencia_binaria', [])[:20]
                grupos_binarios = resultado_ds.get('grupos_binarios', [])[:10]
                palavras_decimais = resultado_ds.get('palavras_decimais', [])[:10]
                histogramas.append({
                    'url_histograma': url_for('static', filename=os.path.basename(resultado_ds.get('caminho_histograma', ''))),
                    'url_sequencia': url_for('static', filename=os.path.basename(resultado_ds.get('caminho_sequencia', ''))),
                    'nome': sinal['nome'],
                    'limiar': f"{resultado_ds.get('limiar', 0):.2f}",
                    'entropia': f"{resultado_ds.get('entropia', 0):.4f}",
                    'sequencia_binaria': sequencia_binaria,
                    'grupos_binarios': grupos_binarios,
                    'palavras_decimais': palavras_decimais,
                    'predicao': predicao,
                    'predicao_cnn': predicao_cnn,
                    'predicao_lstm': predicao_lstm
                })
            except Exception:
                continue


        return render_template(
            "grafico.html",
            graficos_html=graficos_html,
            histogramas=histogramas,
            categoria=categoria
        )
    except Exception as erro_geral:
        return render_template("erro.html", mensagem=f"Erro ao carregar a página principal: {erro_geral}")

def executar_retreinamento_background():
    """Executa o retreinamento em background"""
    global retraining_status, retraining_logs
    retraining_status = "running"
    retraining_logs.clear()
    
    try:
        sucesso = inicializar_classificador()
        if sucesso:
            retraining_status = "completed"
        else:
            retraining_status = "error"
    except Exception as e:
        retraining_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Erro geral: {e}")
        retraining_status = "error"

@app.route("/retreinar", methods=["POST"])
def retreinar():
    """Rota para retreinar o modelo de machine learning."""
    global retraining_status
    
    if retraining_status == "running":
        return jsonify({"status": "running", "message": "Retreinamento já está em andamento"})
    
    # Inicia o retreinamento em background
    thread = threading.Thread(target=executar_retreinamento_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started", "message": "Retreinamento iniciado"})

@app.route("/retreinar_todos", methods=["POST"])
def retreinar_todos():
    """Rota para retreinar todos os modelos de IA."""
    global retraining_status
    
    if retraining_status == "running":
        return jsonify({"status": "running", "message": "Retreinamento já está em andamento"})
    
    # Inicia o retreinamento de todos os modelos em background
    thread = threading.Thread(target=executar_retreinamento_todos_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started", "message": "Retreinamento de todos os modelos iniciado"})

def executar_retreinamento_todos_background():
    """Executa o retreinamento de todos os modelos em background"""
    global retraining_status, retraining_logs
    retraining_status = "running"
    retraining_logs.clear()
    
    try:
        def log_retraining(mensagem):
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {mensagem}"
            retraining_logs.append(log_entry)
            print(log_entry)
        
        log_retraining("🚀 Iniciando retreinamento de todos os modelos...")
        
        # Criar dataset
        log_retraining("📊 Criando dataset de treinamento...")
        X, y, _ = classifier.criar_dataset(limite=20)
        log_retraining(f"✅ Dataset criado: {X.shape[0]} amostras, {X.shape[1]} features")
        
        # Retreinar Random Forest (modelo principal)
        log_retraining("🧠 Retreinando Random Forest...")
        classifier.criar_modelo(tipo_modelo='random_forest')
        classifier.treinar_modelo(X, y)
        classifier.salvar_modelo()
        log_retraining("✅ Random Forest retreinado!")
        
        # Retreinar CNN
        log_retraining("🧠 Retreinando CNN...")
        global classifier_cnn
        classifier_cnn = EEGClassifier()
        classifier_cnn.criar_modelo('mlp_tabular')
        classifier_cnn.treinar_modelo(X, y)
        classifier_cnn.salvar_modelo('modelo_mlp_tabular.pkl')
        log_retraining("✅ MLP Tabular retreinado e salvo!")
        
        # Retreinar LSTM
        log_retraining("🧠 Retreinando LSTM...")
        global classifier_lstm
        classifier_lstm = EEGClassifier()
        classifier_lstm.criar_modelo('lstm')
        classifier_lstm.treinar_modelo(X, y)
        classifier_lstm.salvar_modelo('modelo_lstm.pkl')
        log_retraining("✅ LSTM retreinado e salvo!")
        
        log_retraining("🎉 Todos os modelos foram retreinados com sucesso!")
        retraining_status = "completed"
        
    except Exception as e:
        log_retraining(f"❌ Erro durante retreinamento: {e}")
        retraining_status = "error"

@app.route("/status_retreinamento")
def status_retreinamento():
    """Rota para verificar o status do retreinamento"""
    global retraining_status, retraining_logs
    return jsonify({
        "status": retraining_status,
        "logs": retraining_logs
    })

@app.route("/dashboard")
def dashboard():
    """Dashboard com estatísticas gerais do sistema."""
    try:
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        cursor.execute("SELECT COUNT(*) FROM sinais")
        total_sinais = cursor.fetchone()[0]
        cursor.execute("""
            SELECT COUNT(*) FROM sinais s
            JOIN usuarios u ON s.idusuario = u.id
            WHERE u.possui = 'S'
        """)
        sinais_sim = cursor.fetchone()[0]
        cursor.execute("""
            SELECT COUNT(*) FROM sinais s
            JOIN usuarios u ON s.idusuario = u.id
            WHERE u.possui = 'N'
        """)
        sinais_nao = cursor.fetchone()[0]
        entropias = []
        sinais_recentes = []
        cursor.execute("""
            SELECT s.id, s.nome, u.possui
            FROM sinais s
            JOIN usuarios u ON s.idusuario = u.id
            ORDER BY s.id DESC
            LIMIT 40
        """)
        sinais_amostra = cursor.fetchall()
        for sinal_id, nome, categoria in sinais_amostra:
            try:
                resultado = aplicar_dinamica_simbolica(sinal_id, m=3)
                if resultado and 'entropia' in resultado:
                    entropias.append(resultado['entropia'])
                    predicao = None
                    if classifier.is_trained:
                        try:
                            predicao = classifier.prever_sinal(sinal_id)
                        except Exception:
                            pass
                    sinais_recentes.append({
                        'nome': nome,
                        'categoria': 'Sim' if categoria == 'S' else 'Não',
                        'entropia': resultado['entropia'],
                        'predicao': predicao
                    })
            except Exception:
                continue
        if not entropias:
            entropias = [0.0]
        entropia_media = sum(entropias) / len(entropias) if entropias else 0
        stats = {
            'total_sinais': total_sinais,
            'sinais_sim': sinais_sim,
            'sinais_nao': sinais_nao,
            'entropia_media': entropia_media
        }
        cursor.close()
        conexao.close()
        return render_template("dashboard.html", 
                             stats=stats, 
                             sinais_recentes=sinais_recentes,
                             entropias=entropias)
    except Exception as e:
        return render_template("erro.html", mensagem=f"Erro ao carregar dashboard: {e}")

def executar_testes_background():
    """Executa os testes em background"""
    global test_status, test_logs
    test_status = "running"
    test_logs.clear()
    
    try:
        testador = TestadorSistema()
        resultado = testador.executar_todos_testes()
        test_logs = resultado['logs']
        test_status = "completed"
    except Exception as e:
        test_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Erro geral nos testes: {e}")
        test_status = "error"

@app.route("/executar_testes", methods=["POST"])
def executar_testes():
    """Rota para executar testes do sistema"""
    global test_status
    
    if test_status == "running":
        return jsonify({"status": "running", "message": "Testes já estão em andamento"})
    
    # Inicia os testes em background
    thread = threading.Thread(target=executar_testes_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started", "message": "Testes iniciados"})

@app.route("/status_testes")
def status_testes():
    """Rota para verificar o status dos testes"""
    global test_status, test_logs
    return jsonify({
        "status": test_status,
        "logs": test_logs
    })

@app.route("/testes")
def pagina_testes():
    """Página para visualizar e executar testes"""
    return render_template("testes.html")

@app.route("/comparador")
def pagina_comparador():
    """Página para comparar modelos de rede neural"""
    return render_template("comparador.html")

@app.route("/executar_comparacao", methods=["POST"])
def executar_comparacao():
    """Executa comparação de modelos em background"""
    global test_status, test_logs
    test_status = "running"
    test_logs.clear()
    
    try:
        # Criar comparador
        comparador = ModeloComparador()
        
        # Adicionar modelos
        comparador.adicionar_modelo("Random Forest", "random_forest")
        comparador.adicionar_modelo("MLP", "mlp")
        comparador.adicionar_modelo("CNN", "cnn")
        comparador.adicionar_modelo("LSTM", "lstm")
        comparador.adicionar_modelo("Hybrid CNN-LSTM", "hybrid")
        
        # Criar dataset
        classifier_temp = EEGClassifier()
        X, y, _ = classifier_temp.criar_dataset(limite=50)
        
        if X is not None and len(X) > 0:
            # Treinar modelos
            comparador.treinar_todos_modelos(X, y)
            
            # Avaliar modelos
            metricas = comparador.avaliar_modelos(X, y)
            
            # Gerar visualizações
            comparador.plotar_comparacao_metricas(metricas)
            comparador.plotar_curvas_treino()
            
            # Gerar relatório
            comparador.gerar_relatorio(metricas)
            
            test_status = "completed"
        else:
            test_logs.append("❌ Não foi possível criar dataset")
            test_status = "error"
            
    except Exception as e:
        test_logs.append(f"❌ Erro na comparação: {e}")
        test_status = "error"
    
    return jsonify({"status": "started", "message": "Comparação iniciada"})

def processar_arquivo_eeg(arquivo):
    """
    Processa um arquivo EEG enviado pelo usuário
    """
    try:
        print(f"🔍 Processando arquivo: {arquivo.filename}")
        
        # Verificar se é um arquivo válido
        if arquivo.filename == '':
            return {'erro': 'Nenhum arquivo selecionado'}
        
        if not arquivo.filename.endswith('.txt'):
            return {'erro': 'Apenas arquivos .txt são aceitos'}
        
        # Gerar nome único para o arquivo
        nome_arquivo = f"upload_{uuid.uuid4().hex[:8]}_{arquivo.filename}"
        caminho_arquivo = os.path.join(UPLOAD_FOLDER, nome_arquivo)
        
        # Salvar arquivo
        arquivo.save(caminho_arquivo)
        
        # Ler dados do arquivo
        with open(caminho_arquivo, 'r') as f:
            linhas = f.readlines()
        
        # Processar linhas (remover espaços, quebras de linha, etc.)
        valores = []
        for linha in linhas:
            linha = linha.strip()
            if linha and linha.replace('.', '').replace('-', '').isdigit():
                valores.append(float(linha))
        
        if len(valores) == 0:
            return {'erro': 'Nenhum valor numérico encontrado no arquivo'}
        
        print(f"📊 Valores lidos: {len(valores)} amostras")
            
            # Inserir no banco de dados
            try:
                conexao = obter_conexao_db()
                print("✅ Conexão com banco estabelecida")
            except Exception as e:
                return {'erro': f'Erro de conexão com banco: {str(e)}'}
            cursor = conexao.cursor()
            
            # Criar usuário temporário (sem categoria - usar 'N' como padrão)
            cursor.execute("""
                INSERT INTO usuarios (possui) 
                VALUES (%s) RETURNING id
            """, ('N',))
            id_usuario = cursor.fetchone()[0]
            
            # Criar sinal
            cursor.execute("""
                INSERT INTO sinais (nome, idusuario) 
                VALUES (%s, %s) RETURNING id
            """, (nome_arquivo, id_usuario))
            id_sinal = cursor.fetchone()[0]
            
            # Inserir valores em lote (otimizado)
            valores_para_inserir = [(id_sinal, valor) for valor in valores]
            cursor.executemany("""
                INSERT INTO valores_sinais (idsinal, valor) 
                VALUES (%s, %s)
            """, valores_para_inserir)
            
            conexao.commit()
            cursor.close()
            conexao.close()
            
            # Aplicar dinâmica simbólica
            try:
                print(f"🔧 Aplicando dinâmica simbólica para sinal {id_sinal}")
                resultado_ds = aplicar_dinamica_simbolica(id_sinal, m=3)
                
                if resultado_ds is None:
                    print("❌ Resultado da dinâmica simbólica é None")
                    return {'erro': 'Erro ao processar dinâmica simbólica'}
                
                print(f"✅ Dinâmica simbólica aplicada com sucesso")
                print(f"📊 Resultado DS: {resultado_ds}")
            except Exception as e:
                print(f"❌ Erro na dinâmica simbólica: {str(e)}")
                return {'erro': f'Erro na dinâmica simbólica: {str(e)}'}
            
            # Fazer predição se o modelo estiver treinado
            predicao = None
            if classifier.is_trained:
                try:
                    predicao_raw = classifier.prever_sinal(id_sinal)
                    # Converter tipos NumPy para Python nativos
                    if predicao_raw:
                        predicao = {
                            'classe_predita': str(predicao_raw.get('classe_predita', '')),
                            'probabilidade': float(predicao_raw.get('probabilidade', 0.0))
                        }
                except Exception as e:
                    print(f"Erro na predição: {e}")
            
            # Calcular confiança baseada na entropia
            entropia = float(resultado_ds.get('entropia', 0))
            print(f"🔍 Entropia calculada: {entropia}")
            print(f"🔍 Tipo da entropia: {type(entropia)}")
            
            # Análise de confiança baseada na entropia
            if entropia < 0.3:
                confianca = "Baixa"
                confianca_porcentagem = 30
                confianca_descricao = "Sinal muito previsível, baixa complexidade"
            elif entropia < 0.6:
                confianca = "Média"
                confianca_porcentagem = 60
                confianca_descricao = "Sinal com complexidade moderada"
            elif entropia < 0.8:
                confianca = "Alta"
                confianca_porcentagem = 80
                confianca_descricao = "Sinal com boa complexidade e variabilidade"
            else:
                confianca = "Muito Alta"
                confianca_porcentagem = 95
                confianca_descricao = "Sinal muito complexo e imprevisível"
            
            print(f"✅ Confiança calculada: {confianca} ({confianca_porcentagem}%)")
            print(f"📝 Descrição: {confianca_descricao}")
            
            # Debug dos dados retornados
            print(f"🔍 Dados do resultado_ds:")
            print(f"  - entropia: {resultado_ds.get('entropia')}")
            print(f"  - limiar: {resultado_ds.get('limiar')}")
            print(f"  - sequencia_binaria: {len(resultado_ds.get('sequencia_binaria', []))} elementos")
            print(f"  - grupos_binarios: {len(resultado_ds.get('grupos_binarios', []))} elementos")
            
            # Preparar resultados
            resultados = {
                'id_sinal': int(id_sinal),
                'nome_arquivo': arquivo.filename,
                'total_amostras': int(len(valores)),
                'entropia': entropia,
                'limiar': float(resultado_ds.get('limiar', 0)),
                'confianca': confianca,
                'confianca_porcentagem': confianca_porcentagem,
                'confianca_descricao': confianca_descricao,
                'url_histograma': f"/static/{os.path.basename(resultado_ds.get('caminho_histograma', ''))}",
                'url_sequencia': f"/static/{os.path.basename(resultado_ds.get('caminho_sequencia', ''))}",
                'sequencia_binaria': [int(x) for x in resultado_ds.get('sequencia_binaria', [])[:20]],
                'grupos_binarios': [int(x) for x in resultado_ds.get('grupos_binarios', [])[:10]],
                'predicao': predicao,
                'sucesso': True
            }
            

            
            return resultados
            
        except Exception as e:
            return {'erro': f'Erro ao processar arquivo: {str(e)}'}

def aplicar_dinamica_simbolica_direta(valores, id_sinal, m=3):
    """
    Aplica dinâmica simbólica diretamente aos valores fornecidos
    """
    try:
        import numpy as np
        from collections import Counter
        import matplotlib.pyplot as plt
        import os
        
        # Converter para numpy array
        valores = np.array(valores)
        
        # Calcular limiar (média)
        limiar = np.mean(valores)
        
        # Gerar sequência binária
        sequencia_binaria = ['1' if x >= limiar else '0' for x in valores]
        
        # Gerar grupos deslizantes
        grupos_binarios = [''.join(sequencia_binaria[i:i+m]) for i in range(len(sequencia_binaria)-m+1)]
        
        # Converter para decimal
        palavras_decimais = [int(grupo, 2) for grupo in grupos_binarios]
        
        # Calcular frequências
        contagem = Counter(palavras_decimais)
        total = sum(contagem.values())
        frequencias = {k: v/total for k, v in contagem.items()}
        
        # Calcular entropia
        probabilidades = np.array(list(frequencias.values()))
        probabilidades_filtradas = probabilidades[(probabilidades > 0) & (probabilidades < 1)]
        if len(probabilidades_filtradas) == 0:
            entropia = 0.0
        else:
            probabilidades_norm = probabilidades_filtradas / np.sum(probabilidades_filtradas)
            entropia_bruta = -np.sum(probabilidades_norm * np.log(probabilidades_norm))
            n_simbolos = len(probabilidades_filtradas)
            if n_simbolos > 1:
                entropia_maxima = np.log(n_simbolos)
                entropia = entropia_bruta / entropia_maxima
            else:
                entropia = 0.0
            entropia = max(0.0, min(1.0, entropia))
        
        # Gerar gráficos
        nome_base = f"sinal_{id_sinal}"
        
        # Histograma
        plt.figure(figsize=(12, 6))
        chaves = sorted(frequencias.keys())
        bars = plt.bar(
            range(len(chaves)),
            [frequencias[k] for k in chaves],
            tick_label=[f"{k:03b}" for k in chaves]
        )
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')
        
        plt.xlabel("Grupos Binários (3 bits)")
        plt.ylabel("Frequência Relativa")
        plt.title(f"Distribuição de Padrões - {nome_base}")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        caminho_histograma = f"static/{nome_base}_histograma.png"
        plt.savefig(caminho_histograma, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico da sequência
        plt.figure(figsize=(12, 6))
        plt.plot(valores[:100], 'b-', linewidth=0.5)
        plt.axhline(y=limiar, color='r', linestyle='--', alpha=0.7, label=f'Limiar: {limiar:.2f}')
        plt.xlabel("Amostra")
        plt.ylabel("Amplitude")
        plt.title(f"Sinal EEG - {nome_base}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        caminho_sequencia = f"static/{nome_base}_sequencia.png"
        plt.savefig(caminho_sequencia, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'entropia': entropia,
            'limiar': limiar,
            'sequencia_binaria': sequencia_binaria,
            'grupos_binarios': grupos_binarios,
            'palavras_decimais': palavras_decimais,
            'frequencias': frequencias,
            'caminho_histograma': caminho_histograma,
            'caminho_sequencia': caminho_sequencia
        }
        
    except Exception as e:
        print(f"❌ Erro na dinâmica simbólica: {e}")
        return None

def extrair_features_manualmente(valores):
    """
    Extrai features manualmente dos valores para predição
    """
    try:
        import numpy as np
        
        valores = np.array(valores)
        
        # Features básicas
        features = {
            'entropia_shannon': 0.0,  # Será calculada pela dinâmica simbólica
            'limiar': float(np.mean(valores)),
            'total_amostras': len(valores),
            'total_padroes': 0,  # Será calculado
            'padroes_unicos': 0,  # Será calculado
            'media_valores': float(np.mean(valores)),
            'desvio_padrao': float(np.std(valores)),
            'variancia': float(np.var(valores)),
            'skewness': float(calcular_skewness(valores)),
            'kurtosis': float(calcular_kurtosis(valores)),
            'amplitude': float(np.max(valores) - np.min(valores)),
            'rms': float(np.sqrt(np.mean(valores**2))),
            'proporcao_uns': 0.0,  # Será calculado
            'transicoes': 0,  # Será calculado
            'comprimento_sequencia': len(valores),
            'max_frequencia': 0.0,  # Será calculado
            'min_frequencia': 0.0,  # Será calculado
            'std_frequencias': 0.0,  # Será calculado
            'entropia_frequencias': 0.0  # Será calculado
        }
        
        # Aplicar dinâmica simbólica para completar features
        resultado_ds = aplicar_dinamica_simbolica_direta(valores, "temp", m=3)
        if resultado_ds:
            features.update({
                'entropia_shannon': resultado_ds['entropia'],
                'total_padroes': len(resultado_ds['palavras_decimais']),
                'padroes_unicos': len(resultado_ds['frequencias']),
                'proporcao_uns': resultado_ds['sequencia_binaria'].count('1') / len(resultado_ds['sequencia_binaria']),
                'transicoes': contar_transicoes(resultado_ds['sequencia_binaria']),
                'max_frequencia': max(resultado_ds['frequencias'].values()),
                'min_frequencia': min(resultado_ds['frequencias'].values()),
                'std_frequencias': float(np.std(list(resultado_ds['frequencias'].values()))),
                'entropia_frequencias': calcular_entropia_shannon(list(resultado_ds['frequencias'].values()))
            })
        
        return features
        
    except Exception as e:
        print(f"❌ Erro ao extrair features: {e}")
        return None

def calcular_skewness(data):
    """Calcula o skewness dos dados"""
    import numpy as np
    n = len(data)
    if n < 3:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    
    skewness = (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
    return skewness

def calcular_kurtosis(data):
    """Calcula o kurtosis dos dados"""
    import numpy as np
    n = len(data)
    if n < 4:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    
    kurtosis = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((data - mean) / std) ** 4) - (3 * (n-1)**2 / ((n-2) * (n-3)))
    return kurtosis

def contar_transicoes(sequencia):
    """Conta o número de transições na sequência binária"""
    if len(sequencia) < 2:
        return 0
    
    transicoes = 0
    for i in range(1, len(sequencia)):
        if sequencia[i] != sequencia[i-1]:
            transicoes += 1
    
    return transicoes

def calcular_entropia_shannon(valores):
    """Calcula a entropia de Shannon de uma lista de valores"""
    import numpy as np
    if len(valores) == 0:
        return 0.0
    
    total = sum(valores)
    if total == 0:
        return 0.0
    
    probabilidades = [v/total for v in valores]
    
    entropia = 0.0
    for p in probabilidades:
        if p > 0:
            entropia -= p * np.log2(p)
    
    return entropia

@app.route("/upload")
def pagina_upload():
    """
    Página dedicada para upload de arquivos EEG
    """
    return render_template("upload.html")

@app.route("/upload_eeg", methods=["POST"])
def upload_eeg():
    """
    Rota para upload e processamento de arquivo EEG
    """
    print(f"📥 Recebendo upload: {request.files}")
    
    if 'arquivo' not in request.files:
        print("❌ Nenhum arquivo encontrado na requisição")
        return jsonify({'erro': 'Nenhum arquivo enviado'})
    
    arquivo = request.files['arquivo']
    print(f"📁 Processando arquivo: {arquivo.filename}")
    
    try:
        resultado = processar_arquivo_eeg(arquivo)
        print(f"✅ Processamento concluído: {resultado.get('sucesso', False)}")
        return jsonify(resultado)
    except Exception as e:
        print(f"❌ Erro no processamento: {str(e)}")
        return jsonify({'erro': f'Erro interno: {str(e)}'})

if __name__ == "__main__":
    # Mostrar configurações ao iniciar
    config.print_config()
    
    inicializar_classificador()
    app.run(
        debug=config.FLASK_DEBUG, 
        port=config.FLASK_PORT,
        host=config.FLASK_HOST
    )
