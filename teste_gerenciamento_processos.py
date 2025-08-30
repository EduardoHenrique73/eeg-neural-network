#!/usr/bin/env python3
"""
Script para testar o sistema de gerenciamento de processos
"""

import requests
import time
import json

def testar_gerenciamento_processos():
    """Testa o sistema de gerenciamento de processos"""
    
    base_url = "http://localhost:5000"
    
    print("🧪 Testando Sistema de Gerenciamento de Processos")
    print("=" * 50)
    
    # 1. Verificar status inicial
    print("\n1️⃣ Verificando status inicial...")
    try:
        response = requests.get(f"{base_url}/status_processos")
        if response.status_code == 200:
            data = response.json()
            print(f"   📊 Processos ativos: {data['total']}")
            for processo in data['processos']:
                print(f"   - {processo['nome']} ({processo['tipo']}): {'🟢 Ativo' if processo['ativo'] else '🔴 Inativo'}")
        else:
            print(f"   ❌ Erro ao verificar status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Erro de conexão: {e}")
    
    # 2. Iniciar retreinamento
    print("\n2️⃣ Iniciando retreinamento...")
    try:
        response = requests.post(f"{base_url}/retreinar")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ {data['message']}")
        else:
            print(f"   ❌ Erro ao iniciar retreinamento: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Erro de conexão: {e}")
    
    # 3. Aguardar um pouco
    print("\n3️⃣ Aguardando 3 segundos...")
    time.sleep(3)
    
    # 4. Verificar processos ativos
    print("\n4️⃣ Verificando processos ativos...")
    try:
        response = requests.get(f"{base_url}/status_processos")
        if response.status_code == 200:
            data = response.json()
            print(f"   📊 Processos ativos: {data['total']}")
            for processo in data['processos']:
                print(f"   - {processo['nome']} ({processo['tipo']}): {'🟢 Ativo' if processo['ativo'] else '🔴 Inativo'}")
        else:
            print(f"   ❌ Erro ao verificar status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Erro de conexão: {e}")
    
    # 5. Navegar para página inicial (deve cancelar processos)
    print("\n5️⃣ Navegando para página inicial (deve cancelar processos)...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("   ✅ Página inicial carregada")
        else:
            print(f"   ❌ Erro ao carregar página inicial: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Erro de conexão: {e}")
    
    # 6. Verificar processos após navegação
    print("\n6️⃣ Verificando processos após navegação...")
    try:
        response = requests.get(f"{base_url}/status_processos")
        if response.status_code == 200:
            data = response.json()
            print(f"   📊 Processos ativos: {data['total']}")
            for processo in data['processos']:
                print(f"   - {processo['nome']} ({processo['tipo']}): {'🟢 Ativo' if processo['ativo'] else '🔴 Inativo'}")
        else:
            print(f"   ❌ Erro ao verificar status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Erro de conexão: {e}")
    
    # 7. Navegar para dashboard
    print("\n7️⃣ Navegando para dashboard...")
    try:
        response = requests.get(f"{base_url}/dashboard")
        if response.status_code == 200:
            print("   ✅ Dashboard carregado")
        else:
            print(f"   ❌ Erro ao carregar dashboard: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Erro de conexão: {e}")
    
    # 8. Verificar processos finais
    print("\n8️⃣ Verificando processos finais...")
    try:
        response = requests.get(f"{base_url}/status_processos")
        if response.status_code == 200:
            data = response.json()
            print(f"   📊 Processos ativos: {data['total']}")
            for processo in data['processos']:
                print(f"   - {processo['nome']} ({processo['tipo']}): {'🟢 Ativo' if processo['ativo'] else '🔴 Inativo'}")
        else:
            print(f"   ❌ Erro ao verificar status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Erro de conexão: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Teste concluído!")

if __name__ == "__main__":
    testar_gerenciamento_processos()
