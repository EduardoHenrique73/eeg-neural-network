#!/usr/bin/env python3
"""
Script para testar o sistema de teste de precisão
"""

import sys
import os

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app, calcular_precisao_ia, obter_conexao_db
    print("✅ Módulos importados com sucesso")
    
    # Testar conexão com banco
    try:
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        # Verificar sinais disponíveis
        cursor.execute("""
            SELECT s.id, s.nome, u.possui
            FROM sinais s
            JOIN usuarios u ON s.idusuario = u.id
            ORDER BY s.id DESC
            LIMIT 5
        """)
        sinais = cursor.fetchall()
        print(f"📊 Sinais disponíveis: {len(sinais)}")
        for sinal_id, nome, categoria in sinais:
            print(f"   - ID: {sinal_id}, Nome: {nome}, Categoria: {categoria}")
        
        cursor.close()
        conexao.close()
        
    except Exception as e:
        print(f"❌ Erro na conexão com banco: {e}")
    
    # Testar função de cálculo de precisão
    print("\n🧪 Testando cálculo de precisão...")
    try:
        precisao = calcular_precisao_ia()
        print(f"✅ Precisão calculada: {precisao}")
    except Exception as e:
        print(f"❌ Erro ao calcular precisão: {e}")
    
    # Testar rota de predição
    print("\n🌐 Testando rota de predição...")
    try:
        with app.test_client() as client:
            if sinais:
                sinal_id = sinais[0][0]  # Primeiro sinal
                response = client.get(f'/predicao_sinal/{sinal_id}')
                print(f"   Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.get_json()
                    print(f"   Resposta: {data}")
                else:
                    print(f"   Erro: {response.get_data(as_text=True)}")
            else:
                print("   ⚠️ Nenhum sinal disponível para teste")
    except Exception as e:
        print(f"❌ Erro no teste da rota: {e}")
    
    # Testar rota de estatísticas
    print("\n📈 Testando rota de estatísticas...")
    try:
        with app.test_client() as client:
            response = client.get('/estatisticas_precisao')
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"   Estatísticas: {data}")
            else:
                print(f"   Erro: {response.get_data(as_text=True)}")
    except Exception as e:
        print(f"❌ Erro no teste de estatísticas: {e}")
    
    # Testar página de teste de precisão
    print("\n📄 Testando página de teste de precisão...")
    try:
        with app.test_client() as client:
            response = client.get('/teste_precisao')
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print("   ✅ Página carregou com sucesso")
                html_content = response.get_data(as_text=True)
                if "Teste de Precisão da IA" in html_content:
                    print("   ✅ Conteúdo correto encontrado")
                else:
                    print("   ⚠️ Conteúdo não encontrado")
            else:
                print(f"   ❌ Erro: {response.get_data(as_text=True)}")
    except Exception as e:
        print(f"❌ Erro no teste da página: {e}")
    
    print("\n✅ Teste do sistema de precisão concluído!")
    
except Exception as e:
    print(f"❌ Erro geral: {e}")
