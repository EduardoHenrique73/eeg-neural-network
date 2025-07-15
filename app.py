from flask import Flask, render_template, request, redirect, url_for
import os
import psycopg2
from dinamica_simbolica import aplicar_dinamica_simbolica
from modulo_funcoes import gerar_grafico_interativo
from ml_classifier import EEGClassifier

app = Flask(__name__)

# Instância global do classificador
classifier = EEGClassifier()

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
    try:
        classifier.carregar_modelo()
        if classifier.is_trained:
            return True
    except Exception:
        pass
    try:
        X, y, _ = classifier.criar_dataset(limite=20)
        classifier.criar_modelo(tipo_modelo='random_forest')
        classifier.treinar_modelo(X, y)
        classifier.salvar_modelo()
        return True
    except Exception:
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

@app.route("/retreinar", methods=["POST"])
def retreinar():
    """Rota para retreinar o modelo de machine learning."""
    try:
        sucesso = inicializar_classificador()
        if sucesso:
            return redirect(url_for("home"))
        else:
            return render_template("erro.html", mensagem="Erro ao treinar o modelo.")
    except Exception as e:
        return render_template("erro.html", mensagem=f"Erro ao retreinar: {e}")

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

if __name__ == "__main__":
    inicializar_classificador()
    app.run(debug=True, port=5000)
