#!/usr/bin/env python3
"""
Script para treinar o modelo e testar o sistema de precisão
"""

import sys
import os

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app, inicializar_classificador, classifier
    print("✅ Módulos importados com sucesso")
    
    # Treinar o modelo
    print("\n🧠 Treinando modelo...")
    try:
        sucesso = inicializar_classificador()
        if sucesso:
            print("✅ Modelo treinado com sucesso!")
            
            # Verificar se o modelo está treinado
            if classifier and hasattr(classifier, 'is_trained') and classifier.is_trained:
                print("✅ Modelo está pronto para uso!")
            else:
                print("⚠️ Modelo não está marcado como treinado")
        else:
            print("❌ Falha ao treinar modelo")
    except Exception as e:
        print(f"❌ Erro ao treinar modelo: {e}")
    
    # Testar predição
    print("\n🧪 Testando predição...")
    try:
        with app.test_client() as client:
            # Buscar um sinal para testar
            from app import obter_conexao_db
            conexao = obter_conexao_db()
            cursor = conexao.cursor()
            cursor.execute("SELECT id FROM sinais ORDER BY id DESC LIMIT 1")
            resultado = cursor.fetchone()
            cursor.close()
            conexao.close()
            
            if resultado:
                sinal_id = resultado[0]
                print(f"   Testando predição para sinal ID: {sinal_id}")
                
                response = client.get(f'/predicao_sinal/{sinal_id}')
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.get_json()
                    if data['sucesso']:
                        print(f"   ✅ Predição bem-sucedida: {data['predicao']}")
                    else:
                        print(f"   ❌ Erro na predição: {data['erro']}")
                else:
                    print(f"   ❌ Erro HTTP: {response.get_data(as_text=True)}")
            else:
                print("   ⚠️ Nenhum sinal encontrado para teste")
                
    except Exception as e:
        print(f"❌ Erro no teste de predição: {e}")
    
    print("\n✅ Teste concluído!")
    
except Exception as e:
    print(f"❌ Erro geral: {e}")
