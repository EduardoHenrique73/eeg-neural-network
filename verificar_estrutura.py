#!/usr/bin/env python3
"""
Script para verificar a estrutura real das tabelas no banco de dados
"""

import psycopg2

def verificar_estrutura():
    """Verifica a estrutura real das tabelas"""
    print("🔍 VERIFICANDO ESTRUTURA DAS TABELAS")
    print("=" * 50)
    
    # Conectar ao banco
    conexao = psycopg2.connect(
        dbname="eeg-projeto",
        user="postgres",
        password="EEG@321",
        host="localhost",
        port="5432"
    )
    cursor = conexao.cursor()
    
    try:
        # Listar todas as tabelas
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tabelas = cursor.fetchall()
        
        print("📋 TABELAS ENCONTRADAS:")
        for tabela in tabelas:
            print(f"   - {tabela[0]}")
        
        # Verificar estrutura de cada tabela
        for tabela in tabelas:
            nome_tabela = tabela[0]
            print(f"\n📊 ESTRUTURA DA TABELA '{nome_tabela}':")
            
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{nome_tabela}'
                ORDER BY ordinal_position
            """)
            colunas = cursor.fetchall()
            
            for coluna, tipo, nullable in colunas:
                print(f"   - {coluna}: {tipo} ({'NULL' if nullable == 'YES' else 'NOT NULL'})")
            
            # Contar registros
            cursor.execute(f"SELECT COUNT(*) FROM {nome_tabela}")
            count = cursor.fetchone()[0]
            print(f"   Total de registros: {count}")
            
            # Mostrar alguns exemplos
            if count > 0:
                cursor.execute(f"SELECT * FROM {nome_tabela} LIMIT 3")
                exemplos = cursor.fetchall()
                print(f"   Exemplos:")
                for exemplo in exemplos:
                    print(f"     {exemplo}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cursor.close()
        conexao.close()

if __name__ == "__main__":
    verificar_estrutura() 