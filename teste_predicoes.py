#!/usr/bin/env python3
"""
Script para testar se as predições estão sendo salvas e recuperadas corretamente
"""

import sys
import os

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app, buscar_predicoes_banco, salvar_predicao_banco, obter_conexao_db
    print("✅ Módulos importados com sucesso")
    
    # Testar conexão com banco
    try:
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        # Verificar se a tabela predicoes_ia existe
        cursor.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name = 'predicoes_ia'
        """)
        tabela_existe = cursor.fetchone()[0] > 0
        print(f"✅ Tabela predicoes_ia existe: {tabela_existe}")
        
        if tabela_existe:
            # Verificar quantas predições existem
            cursor.execute("SELECT COUNT(*) FROM predicoes_ia")
            total_predicoes = cursor.fetchone()[0]
            print(f"✅ Total de predições no banco: {total_predicoes}")
            
            # Verificar predições por tipo de modelo
            cursor.execute("""
                SELECT tipo_modelo, COUNT(*) 
                FROM predicoes_ia 
                GROUP BY tipo_modelo
            """)
            predicoes_por_modelo = cursor.fetchall()
            print("📊 Predições por modelo:")
            for modelo, count in predicoes_por_modelo:
                print(f"   - {modelo}: {count}")
            
            # Buscar alguns sinais para testar
            cursor.execute("SELECT id, nome FROM sinais ORDER BY id DESC LIMIT 3")
            sinais_teste = cursor.fetchall()
            
            print("\n🔍 Testando busca de predições:")
            for sinal_id, nome in sinais_teste:
                predicoes = buscar_predicoes_banco(sinal_id)
                print(f"   Sinal {nome} (ID: {sinal_id}):")
                if predicoes:
                    for modelo, dados in predicoes.items():
                        print(f"     - {modelo}: {dados['classe_predita']} ({dados['probabilidade']:.3f})")
                else:
                    print(f"     - Nenhuma predição encontrada")
        
        cursor.close()
        conexao.close()
        
    except Exception as e:
        print(f"❌ Erro na conexão com banco: {e}")
    
    # Testar função de salvar predição
    print("\n🧪 Testando função de salvar predição...")
    try:
        # Criar uma predição de teste
        predicao_teste = {
            'classe_predita': 'Sim',
            'probabilidade': 0.85
        }
        
        # Salvar predição de teste
        salvar_predicao_banco(1, 'teste_modelo', predicao_teste)
        print("✅ Predição de teste salva")
        
        # Buscar predição de teste
        predicoes = buscar_predicoes_banco(1)
        if 'teste_modelo' in predicoes:
            print("✅ Predição de teste recuperada com sucesso")
            print(f"   Dados: {predicoes['teste_modelo']}")
        else:
            print("❌ Predição de teste não foi encontrada")
            
    except Exception as e:
        print(f"❌ Erro no teste de predição: {e}")
    
    print("\n✅ Teste concluído!")
    
except Exception as e:
    print(f"❌ Erro geral: {e}")
