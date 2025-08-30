#!/usr/bin/env python3
"""
Script para testar se a função home está buscando e exibindo as predições corretamente
"""

import sys
import os

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app, buscar_predicoes_banco, obter_conexao_db
    print("✅ Módulos importados com sucesso")
    
    # Testar com alguns sinais específicos
    sinais_teste = [262, 261, 260]  # IDs dos sinais que sabemos que têm predições
    
    print("\n🔍 Testando busca de predições para sinais específicos:")
    for sinal_id in sinais_teste:
        predicoes = buscar_predicoes_banco(sinal_id)
        print(f"\n📊 Sinal ID {sinal_id}:")
        if predicoes:
            for modelo, dados in predicoes.items():
                print(f"   - {modelo}: {dados['classe_predita']} ({dados['probabilidade']:.3f})")
        else:
            print("   - Nenhuma predição encontrada")
    
    # Testar com o cliente Flask
    print("\n🌐 Testando página inicial via Flask:")
    try:
        with app.test_client() as client:
            response = client.get('/?limite=3&categoria=')
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print("   ✅ Página carregou com sucesso")
                # Verificar se há predições no HTML
                html_content = response.get_data(as_text=True)
                if "Clique em 'Retreinar' para executar" in html_content:
                    print("   ⚠️ Ainda mostra placeholders")
                else:
                    print("   ✅ Mostra predições reais")
            else:
                print("   ❌ Erro ao carregar página")
    except Exception as e:
        print(f"   ❌ Erro no teste Flask: {e}")
    
    print("\n✅ Teste concluído!")
    
except Exception as e:
    print(f"❌ Erro geral: {e}")
