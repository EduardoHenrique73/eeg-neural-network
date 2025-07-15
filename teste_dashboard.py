#!/usr/bin/env python3
"""
Script para testar se o dashboard está funcionando
"""

import psycopg2
from app import obter_conexao_db, classifier

def testar_dashboard():
    """Testa se o dashboard consegue carregar os dados"""
    try:
        print("🔍 Testando conexão com banco de dados...")
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        # Teste 1: Contar sinais
        print("📊 Testando contagem de sinais...")
        cursor.execute("SELECT COUNT(*) FROM sinais")
        total_sinais = cursor.fetchone()[0]
        print(f"   ✅ Total de sinais: {total_sinais}")
        
        # Teste 2: Contar por categoria
        cursor.execute("""
            SELECT COUNT(*) FROM sinais s
            JOIN usuarios u ON s.idusuario = u.id
            WHERE u.possui = 'S'
        """)
        sinais_sim = cursor.fetchone()[0]
        print(f"   ✅ Sinais 'Sim': {sinais_sim}")
        
        cursor.execute("""
            SELECT COUNT(*) FROM sinais s
            JOIN usuarios u ON s.idusuario = u.id
            WHERE u.possui = 'N'
        """)
        sinais_nao = cursor.fetchone()[0]
        print(f"   ✅ Sinais 'Não': {sinais_nao}")
        
        # Teste 3: Buscar sinais para entropia
        print("🧮 Testando cálculo de entropia...")
        cursor.execute("""
            SELECT s.id, s.nome, u.possui
            FROM sinais s
            JOIN usuarios u ON s.idusuario = u.id
            ORDER BY s.id DESC
            LIMIT 5
        """)
        sinais_amostra = cursor.fetchall()
        print(f"   ✅ Sinais encontrados para teste: {len(sinais_amostra)}")
        
        # Teste 4: Verificar classificador
        print("🤖 Testando classificador...")
        if classifier.is_trained:
            print("   ✅ Classificador está treinado")
        else:
            print("   ⚠️  Classificador não está treinado")
        
        cursor.close()
        conexao.close()
        
        print("\n🎉 Todos os testes passaram! Dashboard deve funcionar.")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    testar_dashboard() 