#!/usr/bin/env python3
"""
Módulo de configuração para carregar variáveis de ambiente
"""

import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv('config.env')

class Config:
    """Classe de configuração centralizada"""
    
    # Configurações do Banco de Dados
    DB_NAME = os.getenv('DB_NAME', 'eeg-projeto')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'CLOUD2W3J')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    
    # Configurações da Aplicação Flask
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
    FLASK_HOST = os.getenv('FLASK_HOST', '127.0.0.1')
    
    # Configurações de Upload
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', '16777216'))
    
    # Configurações do Modelo ML
    MODEL_PATH = os.getenv('MODEL_PATH', 'modelo_eeg.pkl')
    MODEL_TYPE = os.getenv('MODEL_TYPE', 'random_forest')
    
    # Configurações de Dinâmica Simbólica
    SYMBOLIC_M = int(os.getenv('SYMBOLIC_M', '3'))
    SYMBOLIC_WINDOW_SIZE = int(os.getenv('SYMBOLIC_WINDOW_SIZE', '3'))
    
    # Configurações de Gráficos
    PLOT_WIDTH = int(os.getenv('PLOT_WIDTH', '12'))
    PLOT_HEIGHT = int(os.getenv('PLOT_HEIGHT', '6'))
    PLOT_DPI = int(os.getenv('PLOT_DPI', '300'))
    
    @classmethod
    def get_db_connection_string(cls):
        """Retorna a string de conexão do banco de dados"""
        return {
            'dbname': cls.DB_NAME,
            'user': cls.DB_USER,
            'password': cls.DB_PASSWORD,
            'host': cls.DB_HOST,
            'port': cls.DB_PORT
        }
    
    @classmethod
    def print_config(cls):
        """Imprime as configurações atuais"""
        print("🔧 CONFIGURAÇÕES DO SISTEMA:")
        print("=" * 50)
        print(f"📊 Banco de Dados: {cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}")
        print(f"👤 Usuário DB: {cls.DB_USER}")
        print(f"🌐 Flask: {cls.FLASK_HOST}:{cls.FLASK_PORT}")
        print(f"🐛 Debug: {cls.FLASK_DEBUG}")
        print(f"📁 Modelo: {cls.MODEL_PATH}")
        print(f"🧮 Dinâmica Simbólica (m): {cls.SYMBOLIC_M}")
        print(f"📊 Tamanho Upload: {cls.MAX_CONTENT_LENGTH} bytes")
        print("=" * 50)

# Instância global da configuração
config = Config()
