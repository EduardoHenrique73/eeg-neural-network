#!/usr/bin/env python3
"""
Verifica os sinais no banco de dados
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
import psycopg2

def verificar_sinais():
    """Verifica quantos sinais existem no banco"""
    
    try:
        conexao = psycopg2.connect(**config.get_db_connection_string())
        cursor = conexao.cursor()
        
        # Contar total de sinais
        cursor.execute("SELECT COUNT(*) FROM sinais")
        total_sinais = cursor.fetchone()[0]
        
        # Contar por categoria
        cursor.execute("""
            SELECT u.possui, COUNT(*) 
            FROM sinais s 
            JOIN usuarios u ON s.idusuario = u.id 
            GROUP BY u.possui
        """)
        categorias = cursor.fetchall()
        
        # Listar alguns sinais
        cursor.execute("""
            SELECT s.id, s.nome, u.possui 
            FROM sinais s 
            JOIN usuarios u ON s.idusuario = u.id 
            ORDER BY s.id 
            LIMIT 10
        """)
        amostras = cursor.fetchall()
        
        cursor.close()
        conexao.close()
        
        print("📊 ESTATÍSTICAS DO BANCO DE DADOS")
        print("=" * 40)
        print(f"Total de sinais: {total_sinais}")
        print()
        print("Distribuição por categoria:")
        for categoria, count in categorias:
            print(f"  {categoria}: {count} sinais")
        print()
        print("Primeiros 10 sinais:")
        for id_sinal, nome, categoria in amostras:
            print(f"  ID {id_sinal}: {nome} ({categoria})")
            
    except Exception as e:
        print(f"❌ Erro ao verificar sinais: {e}")

if __name__ == "__main__":
    verificar_sinais()
