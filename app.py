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

app = Flask(__name__)

# Instância global do classificador
classifier = EEGClassifier()

# Variáveis globais para logs e status
retraining_logs = []
retraining_status = "idle"  # idle, running, completed, error
test_logs = []
test_status = "idle"

def obter_conexao_db():
    """Conecta ao banco de dados PostgreSQL."""
    return psycopg2.connect(
        dbname="eeg-projeto",
        user="postgres",
        password="EEG@321",
        host="localhost",
        port="5432"
    )

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
        classifier.carregar_modelo()
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
        classifier.salvar_modelo()
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
        for sinal in resultado['dados_sinais']:
            try:
                resultado_ds = aplicar_dinamica_simbolica(sinal['id'], m=3)
                if resultado_ds is None:
                    continue
                predicao = None
                if classifier.is_trained:
                    try:
                        predicao = classifier.prever_sinal(sinal['id'])
                    except Exception:
                        pass
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
                    'predicao': predicao
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
            LIMIT 10
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

if __name__ == "__main__":
    inicializar_classificador()
    app.run(debug=True, port=5000)
