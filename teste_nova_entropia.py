#!/usr/bin/env python3
"""
Teste da nova implementação da Entropia de Shannon
"""

import numpy as np

def calcular_entropia_shannon(frequencias):
    """
    Calcula a entropia de Shannon normalizada usando logaritmo natural (ln)
    
    ✅ Características:
    - Usa logaritmo natural (ln)
    - Retorna valor normalizado entre 0 e 1
    - Ignora valores de frequência relativa que sejam 0 ou 1
    """
    if not frequencias:
        return 0.0
    
    # Converte as frequências para array numpy
    probabilidades = np.array(list(frequencias.values()))
    
    # Filtra valores: ignora 0 e 1
    probabilidades_filtradas = probabilidades[(probabilidades > 0) & (probabilidades < 1)]
    
    if len(probabilidades_filtradas) == 0:
        return 0.0
    
    # Normaliza as probabilidades filtradas
    probabilidades_norm = probabilidades_filtradas / np.sum(probabilidades_filtradas)
    
    # Calcula entropia usando logaritmo natural (ln)
    # H = -Σ(p_i * ln(p_i))
    entropia_bruta = -np.sum(probabilidades_norm * np.log(probabilidades_norm))
    
    # Normaliza para o intervalo [0, 1]
    # Entropia máxima possível com n símbolos é ln(n)
    n_simbolos = len(probabilidades_filtradas)
    if n_simbolos > 1:
        entropia_maxima = np.log(n_simbolos)
        entropia_normalizada = entropia_bruta / entropia_maxima
    else:
        entropia_normalizada = 0.0
    
    # Garante que está no intervalo [0, 1]
    return max(0.0, min(1.0, entropia_normalizada))

def testar_nova_entropia():
    """Testa a nova implementação da entropia de Shannon"""
    
    print("🧪 TESTANDO NOVA ENTROPIA DE SHANNON")
    print("=" * 50)
    print("✅ Usa logaritmo natural (ln)")
    print("✅ Retorna valor normalizado entre 0 e 1")
    print("✅ Ignora valores 0 e 1")
    print("=" * 50)
    
    # Teste 1: Distribuição uniforme (entropia máxima = 1.0)
    print("\n1️⃣ Teste: Distribuição Uniforme")
    freq_uniforme = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    entropia_uniforme = calcular_entropia_shannon(freq_uniforme)
    print(f"   Frequências: {freq_uniforme}")
    print(f"   Entropia normalizada: {entropia_uniforme:.4f}")
    print(f"   Esperado: 1.0000 (máximo normalizado)")
    
    # Teste 2: Distribuição concentrada (entropia baixa)
    print("\n2️⃣ Teste: Distribuição Concentrada")
    freq_concentrada = {0: 0.9, 1: 0.05, 2: 0.03, 3: 0.02}
    entropia_concentrada = calcular_entropia_shannon(freq_concentrada)
    print(f"   Frequências: {freq_concentrada}")
    print(f"   Entropia normalizada: {entropia_concentrada:.4f}")
    print(f"   Esperado: ~0.3-0.5 (baixa entropia)")
    
    # Teste 3: Caso com valores 0 e 1 (devem ser ignorados)
    print("\n3️⃣ Teste: Com Valores 0 e 1 (Ignorados)")
    freq_com_zeros = {0: 0.0, 1: 0.5, 2: 0.3, 3: 0.2, 4: 1.0}
    entropia_zeros = calcular_entropia_shannon(freq_com_zeros)
    print(f"   Frequências: {freq_com_zeros}")
    print(f"   Entropia normalizada: {entropia_zeros:.4f}")
    print(f"   Baseado apenas em [0.5, 0.3, 0.2]")
    
    # Teste 4: Apenas um símbolo válido
    print("\n4️⃣ Teste: Apenas um Símbolo Válido")
    freq_unico = {0: 0.0, 1: 1.0, 2: 0.5}
    entropia_unico = calcular_entropia_shannon(freq_unico)
    print(f"   Frequências: {freq_unico}")
    print(f"   Entropia normalizada: {entropia_unico:.4f}")
    print(f"   Esperado: 0.0000 (apenas um símbolo válido)")
    
    # Teste 5: Dois símbolos válidos
    print("\n5️⃣ Teste: Dois Símbolos Válidos")
    freq_dois = {0: 0.0, 1: 0.6, 2: 0.4, 3: 1.0}
    entropia_dois = calcular_entropia_shannon(freq_dois)
    print(f"   Frequências: {freq_dois}")
    print(f"   Entropia normalizada: {entropia_dois:.4f}")
    print(f"   Esperado: ~0.9710 (quase máxima para 2 símbolos)")
    
    # Teste 6: Dicionário vazio
    print("\n6️⃣ Teste: Dicionário Vazio")
    freq_vazio = {}
    entropia_vazio = calcular_entropia_shannon(freq_vazio)
    print(f"   Frequências: {freq_vazio}")
    print(f"   Entropia normalizada: {entropia_vazio:.4f}")
    print(f"   Esperado: 0.0000")
    
    print("\n" + "=" * 50)
    print("✅ Testes concluídos!")
    
    # Validação matemática
    print("\n📊 VALIDAÇÃO MATEMÁTICA:")
    print(f"   Entropia máxima (4 símbolos): ln(4) = {np.log(4):.4f}")
    print(f"   Entropia máxima (2 símbolos): ln(2) = {np.log(2):.4f}")
    print(f"   Entropia normalizada uniforme: {entropia_uniforme:.4f}")
    
    # Verificações
    verificacoes = []
    verificacoes.append(0.0 <= entropia_uniforme <= 1.0)
    verificacoes.append(0.0 <= entropia_concentrada <= 1.0)
    verificacoes.append(entropia_unico == 0.0)
    verificacoes.append(entropia_vazio == 0.0)
    
    print(f"\n🔍 VERIFICAÇÕES:")
    print(f"   Todas as entropias estão entre 0 e 1: {all(verificacoes)}")
    print(f"   Entropia uniforme = 1.0: {abs(entropia_uniforme - 1.0) < 0.001}")
    print(f"   Entropia concentrada < uniforme: {entropia_concentrada < entropia_uniforme}")

if __name__ == "__main__":
    testar_nova_entropia() 