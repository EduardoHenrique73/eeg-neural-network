#!/usr/bin/env python3
"""
Script de teste para verificar se o dashboard está funcionando
"""

import sys
import os

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app, dashboard, obter_conexao_db
    print("✅ Módulos importados com sucesso")
    
    # Testar conexão com banco
    try:
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        cursor.execute("SELECT COUNT(*) FROM sinais")
        total = cursor.fetchone()[0]
        cursor.close()
        conexao.close()
        print(f"✅ Conexão com banco OK - Total de sinais: {total}")
    except Exception as e:
        print(f"❌ Erro na conexão com banco: {e}")
    
    # Testar função dashboard
    try:
        with app.test_client() as client:
            response = client.get('/dashboard')
            if response.status_code == 200:
                print("✅ Dashboard funcionando - Status 200")
            else:
                print(f"❌ Dashboard com erro - Status {response.status_code}")
                print(f"Resposta: {response.data.decode()}")
    except Exception as e:
        print(f"❌ Erro ao testar dashboard: {e}")
        
except Exception as e:
    print(f"❌ Erro geral: {e}")
    import traceback
    traceback.print_exc()
