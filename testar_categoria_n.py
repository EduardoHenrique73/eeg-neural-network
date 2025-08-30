import psycopg2
from config import config
from ml_classifier import EEGClassifier

def testar_categoria_n():
    print("🔍 Testando análise de sinais categoria N...")
    
    # Conectar ao banco
    conn = psycopg2.connect(**config.get_db_connection_string())
    cur = conn.cursor()
    
    # Buscar sinais categoria N
    cur.execute("""
        SELECT s.id, s.nome, u.possui 
        FROM sinais s 
        JOIN usuarios u ON s.idusuario = u.id 
        WHERE u.possui = 'N' 
        ORDER BY s.id 
        LIMIT 5
    """)
    sinais_n = cur.fetchall()
    
    print(f"📊 Encontrados {len(sinais_n)} sinais categoria N:")
    for id_sinal, nome, categoria in sinais_n:
        print(f"  - ID: {id_sinal}, Nome: {nome}, Categoria: {categoria}")
    
    # Carregar modelos
    print("\n🧠 Carregando modelos...")
    
    # CNN Original
    cnn = EEGClassifier()
    cnn.carregar_modelo('modelo_cnn.pkl')
    print(f"  - CNN Original: {'✅' if cnn.is_trained else '❌'}")
    
    # MLP Tabular
    mlp = EEGClassifier()
    mlp.carregar_modelo('modelo_mlp_tabular.pkl')
    print(f"  - MLP Tabular: {'✅' if mlp.is_trained else '❌'}")
    
    # LSTM
    lstm = EEGClassifier()
    lstm.carregar_modelo('modelo_lstm.pkl')
    print(f"  - LSTM: {'✅' if lstm.is_trained else '❌'}")
    
    # Random Forest
    rf = EEGClassifier()
    rf.carregar_modelo('modelo_eeg.pkl')
    print(f"  - Random Forest: {'✅' if rf.is_trained else '❌'}")
    
    # Testar predições
    print("\n🔮 Testando predições para sinais categoria N:")
    
    for id_sinal, nome, categoria in sinais_n:
        print(f"\n📊 Sinal: {nome} (ID: {id_sinal}, Categoria Real: {categoria})")
        
        # CNN Original
        try:
            pred_cnn = cnn.prever_sinal(id_sinal)
            if pred_cnn:
                print(f"  - CNN Original: {pred_cnn['classe_predita']} ({pred_cnn['probabilidade']:.3f})")
            else:
                print(f"  - CNN Original: ❌ Erro na predição")
        except Exception as e:
            print(f"  - CNN Original: ❌ Erro: {e}")
        
        # MLP Tabular
        try:
            pred_mlp = mlp.prever_sinal(id_sinal)
            if pred_mlp:
                print(f"  - MLP Tabular: {pred_mlp['classe_predita']} ({pred_mlp['probabilidade']:.3f})")
            else:
                print(f"  - MLP Tabular: ❌ Erro na predição")
        except Exception as e:
            print(f"  - MLP Tabular: ❌ Erro: {e}")
        
        # LSTM
        try:
            pred_lstm = lstm.prever_sinal(id_sinal)
            if pred_lstm:
                print(f"  - LSTM: {pred_lstm['classe_predita']} ({pred_lstm['probabilidade']:.3f})")
            else:
                print(f"  - LSTM: ❌ Erro na predição")
        except Exception as e:
            print(f"  - LSTM: ❌ Erro: {e}")
        
        # Random Forest
        try:
            pred_rf = rf.prever_sinal(id_sinal)
            if pred_rf:
                print(f"  - Random Forest: {pred_rf['classe_predita']} ({pred_rf['probabilidade']:.3f})")
            else:
                print(f"  - Random Forest: ❌ Erro na predição")
        except Exception as e:
            print(f"  - Random Forest: ❌ Erro: {e}")
    
    cur.close()
    conn.close()
    
    print("\n✅ Teste concluído!")

if __name__ == "__main__":
    testar_categoria_n()
