#!/usr/bin/env python3
"""
Script para verificar quantos sinais existem no banco de dados
"""

import sys
import os

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import obter_conexao_db
    print("✅ Módulos importados com sucesso")
    
    # Verificar total de sinais
    try:
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        # Total de sinais
        cursor.execute("SELECT COUNT(*) FROM sinais")
        total_sinais = cursor.fetchone()[0]
        print(f"📊 Total de sinais no banco: {total_sinais}")
        
        # Sinais por categoria
        cursor.execute("""
            SELECT u.possui, COUNT(*) 
            FROM sinais s
            JOIN usuarios u ON s.idusuario = u.id
            GROUP BY u.possui
        """)
        sinais_por_categoria = cursor.fetchall()
        print("📈 Sinais por categoria:")
        for categoria, count in sinais_por_categoria:
            print(f"   - {categoria}: {count}")
        
        # Total de predições atuais
        cursor.execute("SELECT COUNT(*) FROM predicoes_ia")
        total_predicoes = cursor.fetchone()[0]
        print(f"🤖 Total de predições salvas: {total_predicoes}")
        
        # Predições por modelo
        cursor.execute("""
            SELECT tipo_modelo, COUNT(*) 
            FROM predicoes_ia 
            GROUP BY tipo_modelo
        """)
        predicoes_por_modelo = cursor.fetchall()
        print("🧠 Predições por modelo:")
        for modelo, count in predicoes_por_modelo:
            print(f"   - {modelo}: {count}")
        
        # Calcular quantas predições deveriam existir
        predicoes_esperadas = total_sinais * 3  # 3 modelos por sinal
        print(f"🎯 Predições esperadas após retreinamento: {predicoes_esperadas}")
        print(f"📊 Cobertura atual: {total_predicoes}/{predicoes_esperadas} ({total_predicoes/predicoes_esperadas*100:.1f}%)")
        
        cursor.close()
        conexao.close()
        
    except Exception as e:
        print(f"❌ Erro na consulta: {e}")
    
    print("\n✅ Verificação concluída!")
    
except Exception as e:
    print(f"❌ Erro geral: {e}")
