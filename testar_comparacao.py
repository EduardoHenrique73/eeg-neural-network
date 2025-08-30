import psycopg2
from config import config
from ml_classifier import EEGClassifier

def testar_comparacao():
    print("🔍 Testando comparação entre categorias N e S...")
    
    # Conectar ao banco
    conn = psycopg2.connect(**config.get_db_connection_string())
    cur = conn.cursor()
    
    # Buscar sinais categoria S
    cur.execute("""
        SELECT s.id, s.nome, u.possui 
        FROM sinais s 
        JOIN usuarios u ON s.idusuario = u.id 
        WHERE u.possui = 'S' 
        ORDER BY s.id 
        LIMIT 3
    """)
    sinais_s = cur.fetchall()
    
    # Buscar sinais categoria N
    cur.execute("""
        SELECT s.id, s.nome, u.possui 
        FROM sinais s 
        JOIN usuarios u ON s.idusuario = u.id 
        WHERE u.possui = 'N' 
        ORDER BY s.id 
        LIMIT 3
    """)
    sinais_n = cur.fetchall()
    
    print(f"📊 Sinais categoria S: {len(sinais_s)}")
    print(f"📊 Sinais categoria N: {len(sinais_n)}")
    
    # Carregar CNN Original
    cnn = EEGClassifier()
    cnn.carregar_modelo('modelo_cnn.pkl')
    
    print("\n🔮 Comparando predições CNN Original:")
    print("=" * 60)
    
    # Testar sinais S
    print("\n📊 SINAIS CATEGORIA S (Real: Sim):")
    for id_sinal, nome, categoria in sinais_s:
        try:
            pred = cnn.prever_sinal(id_sinal)
            if pred:
                acerto = "✅" if pred['classe_predita'] == 'Sim' else "❌"
                print(f"  {acerto} {nome}: {pred['classe_predita']} ({pred['probabilidade']:.3f})")
            else:
                print(f"  ❌ {nome}: Erro na predição")
        except Exception as e:
            print(f"  ❌ {nome}: Erro - {e}")
    
    # Testar sinais N
    print("\n📊 SINAIS CATEGORIA N (Real: Não):")
    for id_sinal, nome, categoria in sinais_n:
        try:
            pred = cnn.prever_sinal(id_sinal)
            if pred:
                acerto = "✅" if pred['classe_predita'] == 'Não' else "❌"
                print(f"  {acerto} {nome}: {pred['classe_predita']} ({pred['probabilidade']:.3f})")
            else:
                print(f"  ❌ {nome}: Erro na predição")
        except Exception as e:
            print(f"  ❌ {nome}: Erro - {e}")
    
    # Estatísticas
    print("\n📈 ESTATÍSTICAS:")
    print("=" * 60)
    
    total_s = len(sinais_s)
    total_n = len(sinais_n)
    acertos_s = 0
    acertos_n = 0
    
    # Contar acertos S
    for id_sinal, nome, categoria in sinais_s:
        try:
            pred = cnn.prever_sinal(id_sinal)
            if pred and pred['classe_predita'] == 'Sim':
                acertos_s += 1
        except:
            pass
    
    # Contar acertos N
    for id_sinal, nome, categoria in sinais_n:
        try:
            pred = cnn.prever_sinal(id_sinal)
            if pred and pred['classe_predita'] == 'Não':
                acertos_n += 1
        except:
            pass
    
    precisao_s = (acertos_s / total_s) * 100 if total_s > 0 else 0
    precisao_n = (acertos_n / total_n) * 100 if total_n > 0 else 0
    precisao_total = ((acertos_s + acertos_n) / (total_s + total_n)) * 100 if (total_s + total_n) > 0 else 0
    
    print(f"🎯 Precisão categoria S: {precisao_s:.1f}% ({acertos_s}/{total_s})")
    print(f"🎯 Precisão categoria N: {precisao_n:.1f}% ({acertos_n}/{total_n})")
    print(f"🎯 Precisão total: {precisao_total:.1f}% ({(acertos_s + acertos_n)}/{(total_s + total_n)})")
    
    cur.close()
    conn.close()
    
    print("\n✅ Comparação concluída!")

if __name__ == "__main__":
    testar_comparacao()
