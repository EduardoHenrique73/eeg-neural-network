#!/usr/bin/env python3
"""
Script para verificar a estrutura dos dados no banco de dados
"""

import psycopg2
import numpy as np
from config import config

def verificar_dados():
    """Verifica a estrutura dos dados no banco"""
    print("🔍 VERIFICANDO DADOS NO BANCO")
    print("=" * 50)
    
    # Conectar ao banco
    conexao = psycopg2.connect(**config.get_db_connection_string())
    cursor = conexao.cursor()
    
    try:
        # Verificar usuários
        print("\n📊 USUÁRIOS:")
        cursor.execute("SELECT id, possui FROM usuarios ORDER BY id")
        usuarios = cursor.fetchall()
        for user_id, possui in usuarios:
            print(f"   ID {user_id}: Possui: {possui}")
        
        # Verificar sinais
        print(f"\n📈 SINAIS:")
        cursor.execute("""
            SELECT s.id, s.nome, u.possui, COUNT(v.valor) as num_valores
            FROM sinais s
            JOIN usuarios u ON s.idusuario = u.id
            LEFT JOIN valores_sinais v ON s.id = v.idsinal
            GROUP BY s.id, s.nome, u.possui
            ORDER BY s.id
        """)
        sinais = cursor.fetchall()
        
        for sinal_id, nome, possui, num_valores in sinais:
            print(f"   Sinal {sinal_id}: {nome} - Possui: {possui} - Valores: {num_valores}")
        
        # Verificar valores de alguns sinais
        print(f"\n🔢 AMOSTRAS DE VALORES:")
        for sinal_id, nome, possui, num_valores in sinais[:5]:  # Primeiros 5 sinais
            if num_valores > 0:
                cursor.execute("SELECT valor FROM valores_sinais WHERE idsinal = %s LIMIT 10", (sinal_id,))
                valores = [row[0] for row in cursor.fetchall()]
                print(f"   Sinal {sinal_id} ({nome}): {valores}")
            else:
                print(f"   Sinal {sinal_id} ({nome}): SEM VALORES")
        
        # Estatísticas gerais
        print(f"\n📊 ESTATÍSTICAS GERAIS:")
        cursor.execute("SELECT COUNT(*) FROM usuarios")
        total_usuarios = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sinais")
        total_sinais = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM valores_sinais")
        total_valores = cursor.fetchone()[0]
        
        print(f"   Total de usuários: {total_usuarios}")
        print(f"   Total de sinais: {total_sinais}")
        print(f"   Total de valores: {total_valores}")
        
        # Verificar sinais por categoria
        print(f"\n📈 SINAIS POR CATEGORIA:")
        cursor.execute("""
            SELECT u.possui, COUNT(s.id) as num_sinais, 
                   SUM(CASE WHEN v.valor IS NOT NULL THEN 1 ELSE 0 END) as sinais_com_valores
            FROM usuarios u
            LEFT JOIN sinais s ON u.id = s.idusuario
            LEFT JOIN valores_sinais v ON s.id = v.idsinal
            GROUP BY u.possui
        """)
        categorias = cursor.fetchall()
        
        for categoria, num_sinais, sinais_com_valores in categorias:
            print(f"   Categoria '{categoria}': {num_sinais} sinais, {sinais_com_valores} com valores")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cursor.close()
        conexao.close()

if __name__ == "__main__":
    verificar_dados() 