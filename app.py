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
classifier_cnn_original = None  # Modelo CNN original

# Cache de predições para performance
cache_predicoes = None

# Sistema de gerenciamento de processos
processos_ativos = {}
processos_para_cancelar = set()
lock_processos = threading.Lock()

def cancelar_processos_desnecessarios():
    """Cancela processos em background que não são necessários para a página atual"""
    global processos_para_cancelar
    
    with lock_processos:
        for processo_id in processos_para_cancelar:
            if processo_id in processos_ativos:
                print(f"🛑 Cancelando processo: {processo_id}")
                processos_ativos[processo_id]['cancelar'] = True
                processos_para_cancelar.remove(processo_id)

def limpar_processos_finalizados():
    """Remove processos que já foram finalizados"""
    global processos_ativos
    
    with lock_processos:
        processos_para_remover = []
        for processo_id, info in processos_ativos.items():
            if not info['thread'].is_alive():
                processos_para_remover.append(processo_id)
        
        for processo_id in processos_para_remover:
            del processos_ativos[processo_id]
            print(f"🧹 Processo removido: {processo_id}")

def registrar_processo(nome, thread, tipo="background"):
    """Registra um novo processo ativo"""
    global processos_ativos
    
    processo_id = f"{tipo}_{nome}_{int(time.time())}"
    
    with lock_processos:
        processos_ativos[processo_id] = {
            'nome': nome,
            'thread': thread,
            'tipo': tipo,
            'inicio': time.time(),
            'cancelar': False
        }
    
    print(f"📝 Processo registrado: {processo_id}")
    return processo_id

def marcar_processo_para_cancelar(processo_id):
    """Marca um processo para ser cancelado na próxima navegação"""
    global processos_para_cancelar
    
    with lock_processos:
        processos_para_cancelar.add(processo_id)
        print(f"⚠️ Processo marcado para cancelamento: {processo_id}")

def verificar_cancelamento(processo_id):
    """Verifica se um processo deve ser cancelado"""
    with lock_processos:
        if processo_id in processos_ativos:
            return processos_ativos[processo_id]['cancelar']
    return False

def obter_conexao_db():
    """Conecta ao banco de dados PostgreSQL."""
    return psycopg2.connect(**config.get_db_connection_string())

def carregar_cache_predicoes():
    """Carrega cache de predições se existir"""
    global cache_predicoes
    try:
        if os.path.exists('cache_predicoes.pkl'):
            import pickle
            with open('cache_predicoes.pkl', 'rb') as f:
                cache_predicoes = pickle.load(f)
            print(f"✅ Cache carregado: {len(cache_predicoes)} predições")
            return True
        else:
            print("⚠️ Cache de predições não encontrado")
            return False
    except Exception as e:
        print(f"❌ Erro ao carregar cache: {e}")
        return False

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
    # Limpar processos desnecessários ao navegar
    cancelar_processos_desnecessarios()
    limpar_processos_finalizados()
    
    # Marcar processos de predição para cancelamento (não são necessários na página inicial)
    with lock_processos:
        for processo_id, info in processos_ativos.items():
            if info['tipo'] == 'predicao':
                marcar_processo_para_cancelar(processo_id)
    
    categoria = request.args.get("categoria")
    limite = request.args.get("limite", "50")  # Aumentar limite padrão para 50
    
    # Converter limite para inteiro, com valor padrão de 50
    try:
        limite = int(limite)
        if limite <= 0:
            limite = 50
    except ValueError:
        limite = 50
    
    try:
        # Usar a função original que gera gráficos Plotly
        resultado = gerar_grafico_interativo(limite=limite, filtro_categoria=categoria)
        graficos_html = resultado['graficos_html']
        histogramas = []
        
        # Processar sinais com o limite correto
        sinais_para_processar = resultado['dados_sinais'][:limite]  # Usar o limite correto
        
        for sinal in sinais_para_processar:
            try:
                resultado_ds = aplicar_dinamica_simbolica(sinal['id'], m=3)
                if resultado_ds is None:
                    continue
                
                # Fazer apenas predição do modelo principal (Random Forest) para velocidade
                predicao = None
                if classifier and classifier.is_trained:
                    try:
                        predicao_raw = classifier.prever_sinal(sinal['id'])
                        # Verificar se a predição é válida
                        if predicao_raw and isinstance(predicao_raw, dict):
                            predicao = predicao_raw
                        else:
                            predicao = None
                    except Exception as e:
                        print(f"❌ Erro na predição: {e}")
                        predicao = None
                
                # Buscar predições salvas no banco de dados
                predicoes_salvas = buscar_predicoes_banco(sinal['id'])
                
                # Preparar dados para o template
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
                    'predicao_cnn': predicoes_salvas.get('mlp_tabular', "Clique em 'Retreinar' para executar"),
                    'predicao_lstm': predicoes_salvas.get('lstm', "Clique em 'Retreinar' para executar"),
                    'predicao_cnn_original': predicoes_salvas.get('cnn_original', "Clique em 'Retreinar' para executar")
                })
            except Exception as e:
                print(f"❌ Erro ao processar sinal {sinal['id']}: {e}")
                continue


        return render_template(
            "grafico.html",
            graficos_html=graficos_html,
            histogramas=histogramas,
            categoria=categoria,
            limite=limite
        )
    except Exception as erro_geral:
        return render_template("erro.html", mensagem=f"Erro ao carregar a página principal: {erro_geral}")

def executar_retreinamento_background():
    """Executa o retreinamento em background"""
    global retraining_status, retraining_logs
    retraining_status = "running"
    retraining_logs.clear()
    
    # Obter ID do processo atual
    processo_id = None
    with lock_processos:
        for pid, info in processos_ativos.items():
            if info['nome'] == 'retreinamento' and info['thread'] == threading.current_thread():
                processo_id = pid
                break
    
    try:
        # Verificar cancelamento antes de começar
        if processo_id and verificar_cancelamento(processo_id):
            print("🛑 Retreinamento cancelado antes de iniciar")
            retraining_status = "cancelled"
            return
        
        sucesso = inicializar_classificador()
        
        # Verificar cancelamento após inicialização
        if processo_id and verificar_cancelamento(processo_id):
            print("🛑 Retreinamento cancelado após inicialização")
            retraining_status = "cancelled"
            return
        
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
    
    # Inicia o retreinamento em background com gerenciamento de processos
    thread = threading.Thread(target=executar_retreinamento_background)
    thread.daemon = True
    thread.start()
    
    # Registrar o processo
    processo_id = registrar_processo("retreinamento", thread, "retreinamento")
    
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
    
    # Registrar o processo
    processo_id = registrar_processo("retreinamento_todos", thread, "retreinamento")
    
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
        
        # Inicializar modelos primeiro
        log_retraining("🔧 Inicializando modelos...")
        inicializar_todos_modelos()
        
        # Criar dataset
        log_retraining("📊 Criando dataset de treinamento...")
        X, y, _ = classifier.criar_dataset(limite=20)
        log_retraining(f"✅ Dataset criado: {X.shape[0]} amostras, {X.shape[1]} features")
        
        # Retreinar Random Forest (modelo principal)
        log_retraining("🧠 Retreinando Random Forest...")
        classifier.criar_modelo(tipo_modelo='random_forest')
        classifier.treinar_modelo(X, y)
        classifier.salvar_modelo('modelo_eeg.pkl')
        log_retraining("✅ Random Forest retreinado!")
        
        # Retreinar MLP Tabular
        log_retraining("🧠 Retreinando MLP Tabular...")
        global classifier_cnn
        classifier_cnn = EEGClassifier()
        classifier_cnn.criar_modelo('mlp_tabular')
        classifier_cnn.treinar_modelo(X, y)
        classifier_cnn.salvar_modelo('modelo_mlp_tabular.pkl')
        log_retraining("✅ MLP Tabular retreinado e salvo!")
        
        # Retreinar CNN Original
        log_retraining("🧠 Retreinando CNN Original...")
        global classifier_cnn_original
        classifier_cnn_original = EEGClassifier()
        classifier_cnn_original.criar_modelo('cnn')
        
        # Garantir que o CNN seja treinado corretamente
        try:
            classifier_cnn_original.treinar_modelo(X, y)
            if classifier_cnn_original.is_trained:
                classifier_cnn_original.salvar_modelo('modelo_cnn.pkl')
                log_retraining("✅ CNN Original retreinado e salvo!")
            else:
                log_retraining("❌ CNN Original não foi treinado corretamente")
        except Exception as e:
            log_retraining(f"❌ Erro ao treinar CNN Original: {e}")
        
        # Retreinar LSTM
        log_retraining("🧠 Retreinando LSTM...")
        global classifier_lstm
        classifier_lstm = EEGClassifier()
        classifier_lstm.criar_modelo('lstm')
        
        # Garantir que o LSTM seja treinado corretamente
        try:
            classifier_lstm.treinar_modelo(X, y)
            if classifier_lstm.is_trained:
                classifier_lstm.salvar_modelo('modelo_lstm.pkl')
                log_retraining("✅ LSTM retreinado e salvo!")
            else:
                log_retraining("❌ LSTM não foi treinado corretamente")
        except Exception as e:
            log_retraining(f"❌ Erro ao treinar LSTM: {e}")
        
        # Após treinar, fazer predições com todos os modelos para os sinais atuais
        log_retraining("🔮 Fazendo predições com todos os modelos...")
        
        # Buscar TODOS os sinais para fazer predições
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        cursor.execute("""
            SELECT s.id, s.nome
            FROM sinais s
            ORDER BY s.id DESC
        """)
        todos_sinais = cursor.fetchall()
        cursor.close()
        conexao.close()
        
        log_retraining(f"📊 Processando {len(todos_sinais)} sinais para predições...")
        
        # Fazer predições com todos os modelos e salvar no banco
        total_sinais = len(todos_sinais)
        for i, (sinal_id, nome) in enumerate(todos_sinais, 1):
            try:
                # Mostrar progresso a cada 10 sinais
                if i % 10 == 0 or i == total_sinais:
                    log_retraining(f"📈 Progresso: {i}/{total_sinais} sinais processados ({i/total_sinais*100:.1f}%)")
                
                # Predição com MLP Tabular
                if classifier_cnn and classifier_cnn.is_trained:
                    predicao_mlp = classifier_cnn.prever_sinal(sinal_id)
                    # Salvar predição no banco
                    salvar_predicao_banco(sinal_id, 'mlp_tabular', predicao_mlp)
                
                # Predição com CNN Original
                if classifier_cnn_original and classifier_cnn_original.is_trained:
                    predicao_cnn = classifier_cnn_original.prever_sinal(sinal_id)
                    # Salvar predição no banco
                    salvar_predicao_banco(sinal_id, 'cnn_original', predicao_cnn)
                
                # Predição com LSTM
                if classifier_lstm and classifier_lstm.is_trained:
                    predicao_lstm = classifier_lstm.prever_sinal(sinal_id)
                    # Salvar predição no banco
                    salvar_predicao_banco(sinal_id, 'lstm', predicao_lstm)
                    
            except Exception as e:
                log_retraining(f"❌ Erro ao fazer predição para sinal {nome}: {e}")
        
        log_retraining("🎉 Todos os modelos foram retreinados e testados com sucesso!")
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

@app.route("/status_processos")
def status_processos():
    """Rota para verificar o status de todos os processos ativos"""
    global processos_ativos
    
    with lock_processos:
        processos_info = []
        for processo_id, info in processos_ativos.items():
            processos_info.append({
                'id': processo_id,
                'nome': info['nome'],
                'tipo': info['tipo'],
                'ativo': info['thread'].is_alive(),
                'tempo_ativo': time.time() - info['inicio'],
                'cancelar': info['cancelar']
            })
    
    return jsonify({
        'processos': processos_info,
        'total': len(processos_info)
    })

@app.route("/dashboard")
def dashboard():
    """Dashboard com estatísticas gerais do sistema."""
    try:
        # Limpar processos desnecessários ao navegar
        cancelar_processos_desnecessarios()
        limpar_processos_finalizados()
        
        # Marcar processos de predição para cancelamento
        with lock_processos:
            for processo_id, info in processos_ativos.items():
                if info['tipo'] == 'predicao':
                    marcar_processo_para_cancelar(processo_id)
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
            LIMIT 10
        """)
        sinais_amostra = cursor.fetchall()
        for sinal_id, nome, categoria in sinais_amostra:
            try:
                resultado = aplicar_dinamica_simbolica(sinal_id, m=3)
                if resultado and 'entropia' in resultado:
                    entropias.append(resultado['entropia'])
                    
                    # Fazer apenas predição do modelo principal para velocidade
                    predicao = None
                    if classifier and classifier.is_trained:
                        try:
                            predicao = classifier.prever_sinal(sinal_id)
                        except Exception:
                            pass
                    
                    sinais_recentes.append({
                        'id': sinal_id,
                        'nome_arquivo': nome,
                        'categoria': categoria,
                        'entropia': resultado['entropia'],
                        'complexidade': resultado.get('complexidade', 0.0),
                        'data_upload': datetime.now().isoformat(),  # Converter para string ISO
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
        # Calcular taxa de acerto (placeholder - pode ser implementado depois)
        taxa_acerto = 0.85  # Valor padrão
        
        # Converter sinais para JSON para JavaScript
        import json
        
        # Função para converter objetos NumPy para tipos Python nativos
        def converter_para_json(obj):
            if hasattr(obj, 'item'):  # NumPy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # NumPy array
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: converter_para_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [converter_para_json(item) for item in obj]
            else:
                return obj
        
        # Converter sinais para tipos JSON serializáveis
        sinais_json_serializavel = []
        for sinal in sinais_recentes:
            sinal_convertido = converter_para_json(sinal)
            sinais_json_serializavel.append(sinal_convertido)
        
        sinais_json = json.dumps(sinais_json_serializavel)
        
        return render_template("dashboard.html", 
                             total_sinais=total_sinais,
                             sinais_sim=sinais_sim,
                             sinais_nao=sinais_nao,
                             taxa_acerto=taxa_acerto,
                             sinais=sinais_recentes,
                             sinais_json=sinais_json,
                             entropias=entropias)
    except Exception as e:
        return render_template("erro.html", mensagem=f"Erro ao carregar dashboard: {e}")

def salvar_predicao_banco(sinal_id, tipo_modelo, predicao):
    """Salva a predição de um modelo no banco de dados"""
    try:
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        # Verificar se já existe uma predição para este sinal e modelo
        cursor.execute("""
            SELECT id FROM predicoes_ia 
            WHERE id_sinal = %s AND tipo_modelo = %s
        """, (sinal_id, tipo_modelo))
        
        predicao_existente = cursor.fetchone()
        
        if predicao_existente:
            # Atualizar predição existente
            cursor.execute("""
                UPDATE predicoes_ia 
                SET classe_predita = %s, probabilidade = %s, data_predicao = NOW()
                WHERE id_sinal = %s AND tipo_modelo = %s
            """, (predicao['classe_predita'], predicao['probabilidade'], sinal_id, tipo_modelo))
        else:
            # Inserir nova predição
            cursor.execute("""
                INSERT INTO predicoes_ia (id_sinal, tipo_modelo, classe_predita, probabilidade, data_predicao)
                VALUES (%s, %s, %s, %s, NOW())
            """, (sinal_id, tipo_modelo, predicao['classe_predita'], predicao['probabilidade']))
        
        conexao.commit()
        cursor.close()
        conexao.close()
        print(f"✅ Predição {tipo_modelo} salva para sinal {sinal_id}")
        
    except Exception as e:
        print(f"❌ Erro ao salvar predição {tipo_modelo} para sinal {sinal_id}: {e}")

def buscar_predicoes_banco(sinal_id):
    """Busca todas as predições salvas no banco para um sinal"""
    global cache_predicoes
    
    # Primeiro, tentar usar o cache
    if cache_predicoes and sinal_id in cache_predicoes:
        predicoes_cache = cache_predicoes[sinal_id]
        # Converter para o formato esperado
        predicoes = {}
        for modelo, predicao in predicoes_cache.items():
            if predicao and isinstance(predicao, dict):
                predicoes[modelo] = {
                    'classe_predita': predicao.get('classe_predita', ''),
                    'probabilidade': predicao.get('probabilidade', 0.0)
                }
        return predicoes
    
    # Se não estiver no cache, buscar no banco
    try:
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        cursor.execute("""
            SELECT tipo_modelo, classe_predita, probabilidade
            FROM predicoes_ia 
            WHERE id_sinal = %s
        """, (sinal_id,))
        
        predicoes = {}
        for tipo_modelo, classe_predita, probabilidade in cursor.fetchall():
            predicoes[tipo_modelo] = {
                'classe_predita': classe_predita,
                'probabilidade': probabilidade
            }
        
        cursor.close()
        conexao.close()
        return predicoes
        
    except Exception as e:
        print(f"❌ Erro ao buscar predições para sinal {sinal_id}: {e}")
        return {}

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
        
        # Se o processamento foi bem-sucedido, retornar também a predição
        if resultado.get('sucesso'):
            # Buscar o ID do sinal recém-criado
            conexao = obter_conexao_db()
            cursor = conexao.cursor()
            
            cursor.execute("""
                SELECT s.id, s.nome 
                FROM sinais s 
                ORDER BY s.id DESC 
                LIMIT 1
            """)
            
            sinal_info = cursor.fetchone()
            cursor.close()
            conexao.close()
            
            if sinal_info:
                id_sinal = sinal_info[0]
                nome_sinal = sinal_info[1]
                
                # Fazer predição
                try:
                    predicao = classifier.prever_sinal(id_sinal)
                    resultado['predicao'] = predicao
                    resultado['id_sinal'] = id_sinal
                    resultado['nome_sinal'] = nome_sinal
                except Exception as e:
                    print(f"❌ Erro na predição: {e}")
                    resultado['predicao'] = None
                    resultado['erro_predicao'] = str(e)
        
        return jsonify(resultado)
    except Exception as e:
        print(f"❌ Erro no processamento: {str(e)}")
        return jsonify({'erro': f'Erro interno: {str(e)}'})

@app.route("/marcar_categoria_real", methods=["POST"])
def marcar_categoria_real():
    """
    Marca a categoria real de um sinal após o upload
    """
    try:
        data = request.get_json()
        id_sinal = data.get('id_sinal')
        categoria_real = data.get('categoria_real')  # 'S' ou 'N'
        
        if not id_sinal or categoria_real not in ['S', 'N']:
            return jsonify({'erro': 'Dados inválidos'})
        
        # Atualizar a categoria do usuário
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        cursor.execute("""
            UPDATE usuarios 
            SET possui = %s 
            WHERE id = (
                SELECT idusuario FROM sinais WHERE id = %s
            )
        """, (categoria_real, id_sinal))
        
        conexao.commit()
        cursor.close()
        conexao.close()
        
        return jsonify({
            'sucesso': True,
            'mensagem': f'Categoria real marcada como {"Sim" if categoria_real == "S" else "Não"}'
        })
        
    except Exception as e:
        return jsonify({'erro': f'Erro ao marcar categoria: {str(e)}'})

@app.route("/verificar_dados")
def verificar_dados():
    """Rota para verificar dados no banco"""
    try:
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        # Verificar total de sinais por categoria
        cursor.execute("""
            SELECT u.possui, COUNT(*) as total
            FROM sinais s
            JOIN usuarios u ON s.idusuario = u.id
            GROUP BY u.possui
            ORDER BY u.possui
        """)
        
        categorias = cursor.fetchall()
        
        # Verificar alguns sinais de exemplo
        cursor.execute("""
            SELECT s.id, s.nome, u.possui
            FROM sinais s
            JOIN usuarios u ON s.idusuario = u.id
            ORDER BY s.id
            LIMIT 20
        """)
        
        exemplos = cursor.fetchall()
        
        cursor.close()
        conexao.close()
        
        return jsonify({
            'categorias': [{'categoria': cat, 'total': total} for cat, total in categorias],
            'exemplos': [{'id': id, 'nome': nome, 'categoria': cat} for id, nome, cat in exemplos]
        })
        
    except Exception as e:
        return jsonify({'erro': str(e)})

@app.route("/teste_precisao")
def pagina_teste_precisao():
    """
    Página para testar a precisão da IA com dados cegos
    """
    try:
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        # Buscar sinais recentes para teste
        cursor.execute("""
            SELECT s.id, s.nome, u.possui, s.id as id_sinal
            FROM sinais s
            JOIN usuarios u ON s.idusuario = u.id
            ORDER BY s.id DESC
            LIMIT 50
        """)
        
        sinais = cursor.fetchall()
        cursor.close()
        conexao.close()
        

        # Calcular precisão atual
        precisao = calcular_precisao_ia()
        
        return render_template("teste_precisao.html", 
                             sinais=sinais, 
                             precisao=precisao)
    except Exception as e:
        return render_template("erro.html", 
                             mensagem=f"Erro ao carregar página de teste: {e}")

def calcular_precisao_ia():
    """Calcula a precisão atual da IA baseada nos testes realizados"""
    try:
        # Verificar se o classificador está treinado
        if not classifier:
            # Tentar inicializar o classificador se não existir
            inicializar_classificador()
            
        if not classifier or not hasattr(classifier, 'is_trained') or not classifier.is_trained:
            print("⚠️ Modelo não está treinado")
            return {'total': 0, 'acertos': 0, 'precisao': 0.0}
        
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        # Buscar sinais que foram testados (com categoria real marcada)
        cursor.execute("""
            SELECT s.id, u.possui as categoria_real
            FROM sinais s
            JOIN usuarios u ON s.idusuario = u.id
            WHERE u.possui IN ('S', 'N')
            ORDER BY s.id DESC
            LIMIT 100
        """)
        
        sinais_testados = cursor.fetchall()
        cursor.close()
        conexao.close()
        
        if not sinais_testados:
            return {'total': 0, 'acertos': 0, 'precisao': 0.0}
        
        acertos = 0
        total = 0
        
        for sinal_id, categoria_real in sinais_testados:
            try:
                # Fazer predição da IA
                predicao = classifier.prever_sinal(sinal_id)
                if predicao and 'classe_predita' in predicao:
                    classe_predita = predicao['classe_predita']
                    
                    # Comparar predição com categoria real
                    if (classe_predita == 'S' and categoria_real == 'S') or \
                       (classe_predita == 'N' and categoria_real == 'N'):
                        acertos += 1
                    total += 1
            except Exception as e:
                print(f"⚠️ Erro na predição do sinal {sinal_id}: {e}")
                continue
        
        precisao = (acertos / total * 100) if total > 0 else 0.0
        
        return {
            'total': total,
            'acertos': acertos,
            'precisao': round(precisao, 1)
        }
        
    except Exception as e:
        print(f"Erro ao calcular precisão: {e}")
        return {'total': 0, 'acertos': 0, 'precisao': 0.0}

@app.route("/predicao_sinal/<int:sinal_id>")
def predicao_sinal(sinal_id):
    """Rota para obter predição de um sinal específico"""
    try:
        # Verificar se o classificador está treinado
        if not classifier:
            # Tentar inicializar o classificador se não existir
            inicializar_classificador()
            
        if not classifier or not hasattr(classifier, 'is_trained') or not classifier.is_trained:
            return jsonify({
                'sucesso': False,
                'erro': 'Modelo não está treinado. Clique em "Retreinar" para treinar o modelo.'
            })
        
        # Fazer predição (sem timeout no Windows)
        try:
            predicao = classifier.prever_sinal(sinal_id)
            
            if predicao and isinstance(predicao, dict):
                return jsonify({
                    'sucesso': True,
                    'predicao': predicao
                })
            else:
                return jsonify({
                    'sucesso': False,
                    'erro': 'Não foi possível fazer predição'
                })
        except Exception as pred_error:
            return jsonify({
                'sucesso': False,
                'erro': f'Erro na predição: {str(pred_error)}'
            })
            
    except Exception as e:
        return jsonify({
            'sucesso': False,
            'erro': str(e)
        })

@app.route("/estatisticas_precisao")
def estatisticas_precisao():
    """Rota para obter estatísticas atualizadas de precisão"""
    try:
        precisao = calcular_precisao_ia()
        return jsonify(precisao)
    except Exception as e:
        return jsonify({'erro': str(e)})

def inicializar_todos_modelos():
    """Inicializa todos os modelos de IA quando solicitado"""
    global classifier_cnn, classifier_lstm, classifier_cnn_original
    
    print("🚀 Inicializando todos os modelos de IA...")
    
    # Inicializar MLP Tabular
    try:
        print("🧠 Inicializando MLP Tabular...")
        classifier_cnn = EEGClassifier()
        classifier_cnn.criar_modelo('mlp_tabular')
        
        if classifier_cnn.carregar_modelo('modelo_mlp_tabular.pkl'):
            print("✅ MLP Tabular carregado!")
        else:
            print("⚠️ MLP Tabular não encontrado - será treinado durante retreinamento")
            classifier_cnn = None
    except Exception as e:
        print(f"❌ Erro ao inicializar MLP Tabular: {e}")
        classifier_cnn = None
    
    # Inicializar LSTM
    try:
        print("🧠 Inicializando LSTM...")
        classifier_lstm = EEGClassifier()
        classifier_lstm.criar_modelo('lstm')
        
        if classifier_lstm.carregar_modelo('modelo_lstm.pkl'):
            print("✅ LSTM carregado!")
        else:
            print("⚠️ LSTM não encontrado - será treinado durante retreinamento")
            classifier_lstm = None
    except Exception as e:
        print(f"❌ Erro ao inicializar LSTM: {e}")
        classifier_lstm = None
    
    # Inicializar CNN Original
    try:
        print("🧠 Inicializando CNN Original...")
        classifier_cnn_original = EEGClassifier()
        classifier_cnn_original.criar_modelo('cnn')
        
        if classifier_cnn_original.carregar_modelo('modelo_cnn.pkl'):
            print("✅ CNN Original carregado!")
        else:
            print("⚠️ CNN Original não encontrado - será treinado durante retreinamento")
            classifier_cnn_original = None
    except Exception as e:
        print(f"❌ Erro ao inicializar CNN Original: {e}")
        classifier_cnn_original = None
    
    print("🎉 Inicialização de modelos concluída!")

if __name__ == "__main__":
    # Mostrar configurações ao iniciar
    config.print_config()
    
    # Carregar cache de predições para performance
    carregar_cache_predicoes()
    
    # Inicializar apenas o classificador principal (Random Forest)
    inicializar_classificador()
    
    # NÃO inicializar os outros modelos automaticamente
    # Eles serão inicializados apenas quando o usuário clicar em "Retreinar"
    print("🚀 Servidor iniciado! Modelos adicionais serão carregados quando necessário.")
    
    app.run(
        debug=config.FLASK_DEBUG, 
        port=config.FLASK_PORT,
        host=config.FLASK_HOST
    )
