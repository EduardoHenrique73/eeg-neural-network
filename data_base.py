import psycopg2
from config import config

def criar_banco():
    conn = psycopg2.connect(**config.get_db_connection_string())
    cursor = conn.cursor()

    # Criação da tabela de usuários
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usuarios (
        id SERIAL PRIMARY KEY,
        possui CHAR(1) NOT NULL CHECK (possui IN ('S', 'N'))
    )
    """)

    # Criação da tabela de sinais
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sinais (
        id SERIAL PRIMARY KEY,
        nome TEXT NOT NULL,
        idusuario INTEGER NOT NULL,
        FOREIGN KEY (idusuario) REFERENCES usuarios(id)
    )
    """)

    # Criação da tabela de valores dos sinais
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS valores_sinais (
        id SERIAL PRIMARY KEY,
        idsinal INTEGER NOT NULL,
        valor FLOAT NOT NULL,
        FOREIGN KEY (idsinal) REFERENCES sinais(id)
    )
    """)

    # Criação da tabela de predições de IA
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predicoes_ia (
        id SERIAL PRIMARY KEY,
        id_sinal INTEGER NOT NULL,
        tipo_modelo VARCHAR(50) NOT NULL,
        classe_predita VARCHAR(10) NOT NULL,
        probabilidade FLOAT NOT NULL,
        data_predicao TIMESTAMP DEFAULT NOW(),
        FOREIGN KEY (id_sinal) REFERENCES sinais(id),
        UNIQUE(id_sinal, tipo_modelo)
    )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    criar_banco()
