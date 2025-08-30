#!/usr/bin/env python3
"""
Script de otimizações de performance para o sistema EEG
"""

import os
import sys
import psycopg2
import pickle
from datetime import datetime
from config import config

def obter_conexao_db():
    """Conecta ao banco de dados PostgreSQL"""
    return psycopg2.connect(**config.get_db_connection_string())

def limpar_cache_antigo():
    """Remove arquivos de cache antigos"""
    print("🧹 Limpando cache antigo...")
    
    cache_files = [
        'cache_predicoes.pkl',
        'static/graph_cache_*.html'
    ]
    
    for pattern in cache_files:
        if '*' in pattern:
            import glob
            files = glob.glob(pattern)
            for file in files:
                try:
                    os.remove(file)
                    print(f"   ✅ Removido: {file}")
                except:
                    pass
        else:
            try:
                if os.path.exists(pattern):
                    os.remove(pattern)
                    print(f"   ✅ Removido: {pattern}")
            except:
                pass

def criar_indices_banco():
    """Cria índices no banco para melhorar performance"""
    print("📊 Criando índices no banco de dados...")
    
    indices = [
        "CREATE INDEX IF NOT EXISTS idx_valores_sinais_idsinal ON valores_sinais(idsinal)",
        "CREATE INDEX IF NOT EXISTS idx_sinais_idusuario ON sinais(idusuario)",
        "CREATE INDEX IF NOT EXISTS idx_usuarios_possui ON usuarios(possui)",
        "CREATE INDEX IF NOT EXISTS idx_predicoes_ia_id_sinal ON predicoes_ia(id_sinal)",
        "CREATE INDEX IF NOT EXISTS idx_predicoes_ia_tipo_modelo ON predicoes_ia(tipo_modelo)"
    ]
    
    try:
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        for indice in indices:
            try:
                cursor.execute(indice)
                print(f"   ✅ Índice criado")
            except Exception as e:
                print(f"   ⚠️ Erro ao criar índice: {e}")
        
        conexao.commit()
        cursor.close()
        conexao.close()
        print("✅ Índices criados com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro ao criar índices: {e}")

def configurar_matplotlib():
    """Configura Matplotlib para melhor performance"""
    print("🎨 Configurando Matplotlib para performance...")
    
    config_matplotlib = """
# Configurações de performance para Matplotlib
import matplotlib
matplotlib.use('Agg')  # Backend não interativo
matplotlib.rcParams['figure.max_open_warning'] = 0
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 100
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.1
matplotlib.rcParams['figure.figsize'] = [8, 6]
matplotlib.rcParams['font.size'] = 10

import matplotlib.pyplot as plt
plt.ioff()  # Desabilitar modo interativo

# Suprimir avisos
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
"""
    
    # Salvar configuração em arquivo
    with open('matplotlib_config.py', 'w') as f:
        f.write(config_matplotlib)
    
    print("✅ Configuração do Matplotlib salva!")

def criar_cache_predicoes():
    """Cria cache inicial de predições"""
    print("💾 Criando cache de predições...")
    
    try:
        from app import classifier, obter_conexao_db
        
        if not classifier or not classifier.is_trained:
            print("   ⚠️ Modelo não treinado, pulando cache de predições")
            return
        
        conexao = obter_conexao_db()
        cursor = conexao.cursor()
        
        # Buscar todos os sinais
        cursor.execute("SELECT id FROM sinais ORDER BY id")
        sinais = cursor.fetchall()
        
        cache = {}
        total = len(sinais)
        
        print(f"   📊 Processando {total} sinais...")
        
        for i, (sinal_id,) in enumerate(sinais, 1):
            try:
                if i % 10 == 0:
                    print(f"   📈 Progresso: {i}/{total} ({i/total*100:.1f}%)")
                
                predicao = classifier.prever_sinal(sinal_id)
                if predicao:
                    cache[sinal_id] = predicao
                    
            except Exception as e:
                print(f"   ⚠️ Erro no sinal {sinal_id}: {e}")
                continue
        
        # Salvar cache
        with open('cache_predicoes.pkl', 'wb') as f:
            pickle.dump(cache, f)
        
        print(f"✅ Cache criado com {len(cache)} predições!")
        
        cursor.close()
        conexao.close()
        
    except Exception as e:
        print(f"❌ Erro ao criar cache: {e}")

def configurar_variaveis_ambiente():
    """Configura variáveis de ambiente para otimização"""
    print("⚙️ Configurando variáveis de ambiente...")
    
    config_env = """
# Configurações de performance
export PYTHONOPTIMIZE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Configurações do Flask
export FLASK_ENV=production
export FLASK_DEBUG=0
"""
    
    # Salvar em arquivo .env
    with open('.env_performance', 'w') as f:
        f.write(config_env)
    
    print("✅ Variáveis de ambiente configuradas!")

def criar_script_inicio_rapido():
    """Cria script para início rápido do sistema"""
    print("🚀 Criando script de início rápido...")
    
    script = """#!/usr/bin/env python3
# Script de início rápido com otimizações
import os
import sys

# Configurar variáveis de ambiente
os.environ['PYTHONOPTIMIZE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Configurar Matplotlib
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 0

# Importar e executar app
from app import app

if __name__ == "__main__":
    print("🚀 Iniciando sistema com otimizações...")
    app.run(debug=False, host='0.0.0.0', port=5000)
"""
    
    with open('inicio_rapido.py', 'w', encoding='utf-8') as f:
        f.write(script)
    
    print("✅ Script de início rápido criado!")

def main():
    """Executa todas as otimizações"""
    print("🔧 Iniciando otimizações de performance...")
    print("=" * 50)
    
    # 1. Limpar cache antigo
    limpar_cache_antigo()
    print()
    
    # 2. Criar índices no banco
    criar_indices_banco()
    print()
    
    # 3. Configurar Matplotlib
    configurar_matplotlib()
    print()
    
    # 4. Configurar variáveis de ambiente
    configurar_variaveis_ambiente()
    print()
    
    # 5. Criar script de início rápido
    criar_script_inicio_rapido()
    print()
    
    # 6. Criar cache de predições (opcional)
    try:
        criar_cache_predicoes()
    except:
        print("⚠️ Cache de predições não criado (modelo não disponível)")
    print()
    
    print("=" * 50)
    print("✅ Todas as otimizações concluídas!")
    print()
    print("📋 Próximos passos:")
    print("1. Use 'python inicio_rapido.py' para iniciar o sistema")
    print("2. Ou configure as variáveis de ambiente em .env_performance")
    print("3. O sistema agora deve carregar muito mais rápido!")

if __name__ == "__main__":
    main()
