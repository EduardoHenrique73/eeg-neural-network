#!/usr/bin/env python3
"""
Script com otimizações de performance para o projeto EEG
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

def criar_cache_predicoes():
    """Cria cache de predições para todos os sinais"""
    print("🚀 Criando cache de predições...")
    
    try:
        from app import obter_conexao_db, classifier, classifier_cnn, classifier_cnn_original, classifier_lstm
        
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        # Buscar todos os sinais
        cursor.execute("SELECT id, nome FROM sinais ORDER BY id")
        todos_sinais = cursor.fetchall()
        cursor.close()
        conexao.close()
        
        cache_predicoes = {}
        total_sinais = len(todos_sinais)
        
        print(f"📊 Processando {total_sinais} sinais...")
        
        for i, (sinal_id, nome) in enumerate(todos_sinais, 1):
            if i % 10 == 0:
                print(f"📈 Progresso: {i}/{total_sinais} ({i/total_sinais*100:.1f}%)")
            
            predicoes_sinal = {}
            
            # Random Forest
            try:
                if classifier and classifier.is_trained:
                    predicao = classifier.prever_sinal(sinal_id)
                    predicoes_sinal['random_forest'] = predicao
            except Exception as e:
                print(f"❌ Erro RF sinal {nome}: {e}")
            
            # MLP Tabular
            try:
                if classifier_cnn and classifier_cnn.is_trained:
                    predicao = classifier_cnn.prever_sinal(sinal_id)
                    predicoes_sinal['mlp_tabular'] = predicao
            except Exception as e:
                print(f"❌ Erro MLP sinal {nome}: {e}")
            
            # CNN Original
            try:
                if classifier_cnn_original and classifier_cnn_original.is_trained:
                    predicao = classifier_cnn_original.prever_sinal(sinal_id)
                    predicoes_sinal['cnn_original'] = predicao
            except Exception as e:
                print(f"❌ Erro CNN sinal {nome}: {e}")
            
            # LSTM
            try:
                if classifier_lstm and classifier_lstm.is_trained:
                    predicao = classifier_lstm.prever_sinal(sinal_id)
                    predicoes_sinal['lstm'] = predicao
            except Exception as e:
                print(f"❌ Erro LSTM sinal {nome}: {e}")
            
            cache_predicoes[sinal_id] = predicoes_sinal
        
        # Salvar cache
        with open('cache_predicoes.pkl', 'wb') as f:
            pickle.dump(cache_predicoes, f)
        
        print(f"✅ Cache salvo com {len(cache_predicoes)} sinais")
        return cache_predicoes
        
    except Exception as e:
        print(f"❌ Erro ao criar cache: {e}")
        return None

def otimizar_banco_dados():
    """Otimiza o banco de dados com índices"""
    print("🔧 Otimizando banco de dados...")
    
    try:
        from app import obter_conexao_db
        
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        # Criar índices para melhorar performance
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_sinais_id ON sinais(id)",
            "CREATE INDEX IF NOT EXISTS idx_sinais_nome ON sinais(nome)",
            "CREATE INDEX IF NOT EXISTS idx_valores_sinais_idsinal ON valores_sinais(idsinal)",
            "CREATE INDEX IF NOT EXISTS idx_predicoes_ia_id_sinal ON predicoes_ia(id_sinal)",
            "CREATE INDEX IF NOT EXISTS idx_predicoes_ia_tipo_modelo ON predicoes_ia(tipo_modelo)",
            "CREATE INDEX IF NOT EXISTS idx_usuarios_possui ON usuarios(possui)"
        ]
        
        for indice in indices:
            try:
                cursor.execute(indice)
                print(f"✅ Índice criado")
            except Exception as e:
                print(f"⚠️ Índice já existe ou erro: {e}")
        
        conexao.commit()
        cursor.close()
        conexao.close()
        
        print("✅ Banco de dados otimizado!")
        
    except Exception as e:
        print(f"❌ Erro ao otimizar banco: {e}")

def limpar_cache_antigo():
    """Remove arquivos de cache antigos"""
    print("🧹 Limpando cache antigo...")
    
    arquivos_cache = [
        'static/graph_cache_*.html',
        'cache_predicoes.pkl',
        '*.pyc',
        '__pycache__'
    ]
    
    for padrao in arquivos_cache:
        try:
            if '*' in padrao:
                import glob
                arquivos = glob.glob(padrao)
                for arquivo in arquivos:
                    if os.path.isfile(arquivo):
                        os.remove(arquivo)
                        print(f"🗑️ Removido: {arquivo}")
            else:
                if os.path.exists(padrao):
                    if os.path.isfile(padrao):
                        os.remove(padrao)
                    elif os.path.isdir(padrao):
                        import shutil
                        shutil.rmtree(padrao)
                    print(f"🗑️ Removido: {padrao}")
        except Exception as e:
            print(f"⚠️ Erro ao remover {padrao}: {e}")
    
    print("✅ Cache limpo!")

def configurar_matplotlib_otimizado():
    """Configura Matplotlib para melhor performance"""
    print("📊 Configurando Matplotlib otimizado...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Backend não-interativo
        matplotlib.rcParams['figure.dpi'] = 100
        matplotlib.rcParams['savefig.bbox'] = 'tight'
        matplotlib.rcParams['figure.max_open_warning'] = 0
        matplotlib.rcParams['savefig.format'] = 'png'
        matplotlib.rcParams['savefig.transparent'] = False
        
        print("✅ Matplotlib configurado para performance!")
        
    except Exception as e:
        print(f"❌ Erro ao configurar Matplotlib: {e}")

def criar_script_inicializacao_rapida():
    """Cria script para inicialização rápida"""
    print("⚡ Criando script de inicialização rápida...")
    
    script_content = '''#!/usr/bin/env python3
"""
Script de inicialização rápida com otimizações
"""

import os
import sys
import pickle
from pathlib import Path

# Configurar variáveis de ambiente para performance
os.environ['PYTHONOPTIMIZE'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'  # Ajustar conforme CPU
os.environ['MKL_NUM_THREADS'] = '4'

# Configurar Matplotlib
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['figure.max_open_warning'] = 0

# Carregar cache de predições se existir
cache_predicoes = None
if os.path.exists('cache_predicoes.pkl'):
    try:
        with open('cache_predicoes.pkl', 'rb') as f:
            cache_predicoes = pickle.load(f)
        print(f"✅ Cache carregado: {len(cache_predicoes)} predições")
    except Exception as e:
        print(f"⚠️ Erro ao carregar cache: {e}")

# Inicializar app
from app import app, inicializar_classificador

if __name__ == "__main__":
    print("🚀 Inicializando servidor otimizado...")
    inicializar_classificador()
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
'''
    
    with open('app_rapido.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ Script de inicialização rápida criado: app_rapido.py")

def main():
    """Executa todas as otimizações"""
    print("🚀 Iniciando otimizações de performance...")
    
    # 1. Limpar cache antigo
    limpar_cache_antigo()
    
    # 2. Configurar Matplotlib
    configurar_matplotlib_otimizado()
    
    # 3. Otimizar banco de dados
    otimizar_banco_dados()
    
    # 4. Criar cache de predições (opcional - demora mais)
    print("\n🤔 Deseja criar cache de predições? (pode demorar alguns minutos)")
    resposta = input("Digite 's' para sim ou Enter para pular: ").lower().strip()
    
    if resposta == 's':
        criar_cache_predicoes()
    
    # 5. Criar script de inicialização rápida
    criar_script_inicializacao_rapida()
    
    print("\n🎉 Otimizações concluídas!")
    print("\n📋 Como usar:")
    print("1. Para inicialização rápida: python app_rapido.py")
    print("2. Cache de predições será usado automaticamente")
    print("3. Banco de dados otimizado com índices")
    print("4. Matplotlib configurado para performance")

if __name__ == "__main__":
    main()
