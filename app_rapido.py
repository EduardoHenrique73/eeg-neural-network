#!/usr/bin/env python3
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
