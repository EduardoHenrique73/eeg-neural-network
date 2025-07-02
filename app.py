from flask import Flask, render_template, request, redirect, url_for
import os
import psycopg2
from dinamica_simbolica import aplicar_dinamica_simbolica
from modulo_funcoes import gerar_grafico_interativo

app = Flask(__name__)

# Função para obter conexão com o banco
def obter_conexao_db():
    return psycopg2.connect(
        dbname="eeg-projeto",
        user="postgres",
        password="EEG@321",
        host="localhost",
        port="5432"
    )

# Página principal
@app.route("/")
def home():
    categoria = request.args.get("categoria")

    try:
        # Obtém gráficos interativos e dados dos sinais
        resultado = gerar_grafico_interativo(limite=10, filtro_categoria=categoria)
        graficos_html = resultado['graficos_html']

        histogramas = []
        for sinal in resultado['dados_sinais']:
            try:
                # Aplica a dinâmica simbólica
                resultado_ds = aplicar_dinamica_simbolica(sinal['id'], m=3)

                histogramas.append({
                    'url_histograma': url_for('static', filename=os.path.basename(resultado_ds['caminho_histograma'])),
                    'url_sequencia': url_for('static', filename=os.path.basename(resultado_ds['caminho_sequencia'])),
                    'nome': sinal['nome'],
                    'limiar': f"{resultado_ds['limiar']:.2f}",
                    'sequencia_binaria': resultado_ds['sequencia_binaria'][:20],  # Mostra só os 20 primeiros
                    'grupos_binarios': resultado_ds['grupos_binarios'][:10],       # Mostra só 10 grupos
                    'palavras_decimais': resultado_ds['palavras_decimais'][:10]    # Mostra só 10 palavras
                })

            except Exception as erro_sinal:
                print(f"[ERRO] Problema ao processar sinal {sinal['nome']} (ID {sinal['id']}): {erro_sinal}")

        # Renderiza o template passando todos os dados
        return render_template(
            "grafico.html",
            graficos_html=graficos_html,
            histogramas=histogramas,
            categoria=categoria
        )

    except Exception as erro_geral:
        print(f"[ERRO GERAL] {erro_geral}")
        return render_template("erro.html", mensagem=f"Erro ao carregar a página principal: {erro_geral}")

# Rota para filtrar categoria via POST
@app.route("/filtrar", methods=["POST"])
def filtrar():
    categoria = request.form.get("categoria")
    if categoria not in ["sim", "nao", None, ""]:
        return render_template("erro.html", mensagem="Categoria inválida.")
    return redirect(url_for("home", categoria=categoria))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
