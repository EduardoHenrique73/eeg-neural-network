#!/usr/bin/env python3
"""
Script para verificar quantos sinais válidos temos para treinar o modelo
"""

import psycopg2
from dinamica_simbolica import aplicar_dinamica_simbolica

def verificar_sinais_validos():
    """Verifica quantos sinais são válidos para treinar o modelo"""
    print("🔍 VERIFICANDO SINAIS VÁLIDOS")
    print("=" * 50)
    
    # Conectar ao banco
    conexao = psycopg2.connect(
        dbname="eeg-projeto",
        user="postgres",
        password="EEG@321",
        host="localhost",
        port="5432"
    )
    cursor = conexao.cursor()
    
    try:
        # Buscar sinais por categoria
        for categoria, label in [('S', 1), ('N', 0)]:
            print(f"\n📊 CATEGORIA '{categoria}' (label: {label}):")
            
            cursor.execute("""
                SELECT s.id, s.nome, u.possui, COUNT(v.valor) as num_valores
                FROM sinais s
                JOIN usuarios u ON s.idusuario = u.id
                LEFT JOIN valores_sinais v ON s.id = v.idsinal
                WHERE u.possui = %s
                GROUP BY s.id, s.nome, u.possui
                ORDER BY s.id
            """, (categoria,))
            
            sinais = cursor.fetchall()
            print(f"   Total de sinais na categoria: {len(sinais)}")
            
            sinais_validos = 0
            sinais_invalidos = 0
            
            for sinal_id, nome, possui, num_valores in sinais:
                print(f"     Testando sinal {sinal_id} ({nome}): {num_valores} valores")
                
                if num_valores == 0:
                    print(f"       ❌ Sem valores no banco")
                    sinais_invalidos += 1
                    continue
                
                try:
                    resultado = aplicar_dinamica_simbolica(sinal_id, m=3)
                    
                    if resultado and len(resultado['sequencia_binaria']) > 0:
                        print(f"       ✅ Válido - Entropia: {resultado['entropia']:.4f}")
                        sinais_validos += 1
                    else:
                        print(f"       ❌ Sequência binária vazia")
                        sinais_invalidos += 1
                        
                except Exception as e:
                    print(f"       ❌ Erro: {e}")
                    sinais_invalidos += 1
            
            print(f"   Resumo categoria '{categoria}':")
            print(f"     Válidos: {sinais_validos}")
            print(f"     Inválidos: {sinais_invalidos}")
            print(f"     Taxa de sucesso: {sinais_validos/(sinais_validos+sinais_invalidos)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cursor.close()
        conexao.close()

if __name__ == "__main__":
    verificar_sinais_validos() 