import os
import psycopg2
import plotly.graph_objs as go
import plotly.io as pio
from config import config

# Configurações do banco
def obter_conexao_db():
    """Cria uma conexão segura com o banco de dados PostgreSQL."""
    return psycopg2.connect(**config.get_db_connection_string())

# Inserir usuário
def inserir_usuario(possui):
    with obter_conexao_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO usuarios (possui) VALUES (%s) RETURNING id",
                (possui,)
            )
            return cur.fetchone()[0]

# Inserir sinal
def inserir_sinal(nome, id_usuario):
    with obter_conexao_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sinais (nome, idusuario) VALUES (%s, %s) RETURNING id",
                (nome, id_usuario)
            )
            return cur.fetchone()[0]

# Inserir vários valores de sinal de uma vez
def inserir_valores_sinal(id_sinal, valores):
    with obter_conexao_db() as conn:
        with conn.cursor() as cur:
            registros = [(id_sinal, valor) for valor in valores]
            cur.executemany(
                "INSERT INTO valores_sinais (idsinal, valor) VALUES (%s, %s)",
                registros
            )

# Processa todos os arquivos .txt e insere no banco
def processar_arquivos():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sinais EEG")

    for categoria in ["sim", "nao"]:
        pasta = os.path.join(base_dir, categoria)
        if not os.path.exists(pasta):
            print(f"[AVISO] Pasta não encontrada: {pasta}")
            continue

        for arquivo_nome in sorted(os.listdir(pasta)):
            if not arquivo_nome.endswith(".txt"):
                continue

            try:
                caminho = os.path.join(pasta, arquivo_nome)
                with open(caminho, "r", encoding="utf-8") as f:
                    dados = f.read()

                id_usuario = inserir_usuario('S' if categoria == "sim" else 'N')
                id_sinal = inserir_sinal(arquivo_nome, id_usuario)
                valores = [float(val) for val in dados.split()]
                inserir_valores_sinal(id_sinal, valores)

                print(f"[OK] Inserido: {arquivo_nome} - Categoria: {categoria.upper()}")

            except Exception as e:
                print(f"[ERRO] Falha ao inserir {arquivo_nome}: {str(e)}")

# Gera gráficos interativos e retorna HTML
def gerar_grafico_interativo(limite=10, filtro_categoria=None):
    with obter_conexao_db() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT s.id, s.nome, array_agg(vs.valor ORDER BY vs.id), u.possui
                FROM sinais s
                JOIN usuarios u ON s.idusuario = u.id
                JOIN valores_sinais vs ON s.id = vs.idsinal
                {where}
                GROUP BY s.id, s.nome, u.possui
                LIMIT %s
            """

            params = []
            where = ""
            if filtro_categoria:
                where = "WHERE u.possui = %s"
                params.append(filtro_categoria.upper())

            cur.execute(query.format(where=where), (*params, limite))
            sinais = cur.fetchall()

    if not sinais:
        return {'graficos_html': [], 'dados_sinais': []}

    htmls = []
    dados_sinais = []

    for id_sinal, nome, valores, possui in sinais:
        try:
            # Verificar se o gráfico já existe no cache
            cache_file = f"static/graph_cache_{id_sinal}.html"
            if os.path.exists(cache_file):
                # Carregar gráfico do cache
                with open(cache_file, 'r', encoding='utf-8') as f:
                    grafico_html = f.read()
                print(f"📁 Gráfico carregado do cache: {nome}")
            else:
                # Gerar novo gráfico
                fig = go.Figure(
                    data=[go.Scatter(x=list(range(len(valores))), y=valores, mode="lines")]
                )
                fig.update_layout(
                    title=f"Sinal EEG: {nome}",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                grafico_html = pio.to_html(
                    fig,
                    include_plotlyjs=True,
                    full_html=False,
                    div_id=f'graph_{id_sinal}'
                )
                
                # Salvar no cache
                os.makedirs("static", exist_ok=True)
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(grafico_html)
                print(f"💾 Gráfico salvo no cache: {nome}")

            htmls.append(f'<div class="graph-container" id="container_{id_sinal}">{grafico_html}</div>')

            dados_sinais.append({
                'id': id_sinal,
                'nome': nome,
                'possui': possui
            })

        except Exception as e:
            print(f"[ERRO] Problema ao gerar gráfico para {nome}: {str(e)}")

    return {
        'graficos_html': htmls,
        'dados_sinais': dados_sinais
    }

# Se rodar direto
if __name__ == "__main__":
    processar_arquivos()
