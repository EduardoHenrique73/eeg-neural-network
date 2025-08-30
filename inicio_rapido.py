#!/usr/bin/env python3
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
