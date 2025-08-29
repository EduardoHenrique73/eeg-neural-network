import psycopg2

def criar_banco():
    conn = psycopg2.connect(
        dbname="eeg-projeto",
        user="postgres",
        password="EEG@321",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    # Criação da tabela de usuários
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usuarios (
        id SERIAL PRIMARY KEY,
        possui CHAR(1) NOT NULL CHECK (possui IN ('S', 'N', 'U'))
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

    conn.commit()
    conn.close()

if __name__ == "__main__":
    criar_banco()
