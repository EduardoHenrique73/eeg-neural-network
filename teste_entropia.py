#!/usr/bin/env python3
"""
Teste da implementação da Entropia de Shannon
"""

import numpy as np
from dinamica_simbolica import calcular_entropia_shannon

def testar_entropia_shannon():
    """Testa a função de entropia de Shannon com diferentes cenários"""
    
    print("🧪 TESTANDO ENTROPIA DE SHANNON")
    print("=" * 50)
    
    # Teste 1: Distribuição uniforme (entropia máxima)
    print("\n1️⃣ Teste: Distribuição Uniforme")
    freq_uniforme = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    entropia_uniforme = calcular_entropia_shannon(freq_uniforme)
    print(f"   Frequências: {freq_uniforme}")
    print(f"   Entropia: {entropia_uniforme:.4f} bits")
    print(f"   Esperado: 2.0000 bits (máximo para 4 símbolos)")
    
    # Teste 2: Distribuição concentrada (entropia baixa)
    print("\n2️⃣ Teste: Distribuição Concentrada")
    freq_concentrada = {0: 0.9, 1: 0.05, 2: 0.03, 3: 0.02}
    entropia_concentrada = calcular_entropia_shannon(freq_concentrada)
    print(f"   Frequências: {freq_concentrada}")
    print(f"   Entropia: {entropia_concentrada:.4f} bits")
    print(f"   Esperado: ~0.5 bits (baixa entropia)")
    
    # Teste 3: Distribuição intermediária
    print("\n3️⃣ Teste: Distribuição Intermediária")
    freq_intermedia = {0: 0.5, 1: 0.3, 2: 0.15, 3: 0.05}
    entropia_intermedia = calcular_entropia_shannon(freq_intermedia)
    print(f"   Frequências: {freq_intermedia}")
    print(f"   Entropia: {entropia_intermedia:.4f} bits")
    print(f"   Esperado: ~1.5 bits (entropia média)")
    
    # Teste 4: Caso extremo - apenas um símbolo
    print("\n4️⃣ Teste: Apenas um Símbolo")
    freq_unico = {0: 1.0}
    entropia_unico = calcular_entropia_shannon(freq_unico)
    print(f"   Frequências: {freq_unico}")
    print(f"   Entropia: {entropia_unico:.4f} bits")
    print(f"   Esperado: 0.0000 bits (entropia mínima)")
    
    # Teste 5: Dicionário vazio
    print("\n5️⃣ Teste: Dicionário Vazio")
    freq_vazio = {}
    entropia_vazio = calcular_entropia_shannon(freq_vazio)
    print(f"   Frequências: {freq_vazio}")
    print(f"   Entropia: {entropia_vazio:.4f} bits")
    print(f"   Esperado: 0.0000 bits")
    
    print("\n" + "=" * 50)
    print("✅ Testes concluídos!")
    
    # Validação matemática
    print("\n📊 VALIDAÇÃO MATEMÁTICA:")
    print(f"   Entropia máxima (4 símbolos): {np.log2(4):.4f} bits")
    print(f"   Entropia mínima: 0.0000 bits")
    print(f"   Entropia uniforme calculada: {entropia_uniforme:.4f} bits")
    
    if abs(entropia_uniforme - 2.0) < 0.001:
        print("   ✅ Entropia uniforme CORRETA!")
    else:
        print("   ❌ Entropia uniforme INCORRETA!")
    
    if entropia_unico == 0.0:
        print("   ✅ Entropia mínima CORRETA!")
    else:
        print("   ❌ Entropia mínima INCORRETA!")

if __name__ == "__main__":
    testar_entropia_shannon() 