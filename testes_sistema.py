import os
import time
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime
from dinamica_simbolica import aplicar_dinamica_simbolica
from ml_classifier import EEGClassifier
from modulo_funcoes import gerar_grafico_interativo

class TestadorSistema:
    """Classe para executar testes completos no sistema EEG"""
    
    def __init__(self):
        self.logs = []
        self.resultados = {}
        
    def log(self, mensagem, tipo="INFO"):
        """Adiciona uma mensagem ao log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {tipo}: {mensagem}"
        self.logs.append(log_entry)
        print(log_entry)
        
    def obter_conexao_db(self):
        """Conecta ao banco de dados PostgreSQL"""
        return psycopg2.connect(
            dbname="eeg-projeto",
            user="postgres",
            password="EEG@321",
            host="localhost",
            port="5432"
        )
    
    def testar_conexao_banco(self):
        """Testa a conexão com o banco de dados"""
        self.log("🔍 Testando conexão com banco de dados...")
        try:
            conexao = self.obter_conexao_db()
            cursor = conexao.cursor()
            
            # Teste básico de conexão
            cursor.execute("SELECT COUNT(*) FROM sinais")
            total_sinais = cursor.fetchone()[0]
            
            # Verificar estrutura das tabelas
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tabelas = [row[0] for row in cursor.fetchall()]
            
            # Contar por categoria
            cursor.execute("""
                SELECT u.possui, COUNT(*) as total
                FROM sinais s
                JOIN usuarios u ON s.idusuario = u.id
                GROUP BY u.possui
                ORDER BY u.possui
            """)
            categorias = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.close()
            conexao.close()
            
            self.log(f"✅ Conexão com banco OK")
            self.log(f"   Total de sinais: {total_sinais}")
            self.log(f"   Tabelas encontradas: {', '.join(tabelas)}")
            for categoria, total in categorias.items():
                self.log(f"   Categoria '{categoria}': {total} sinais")
            
            return {
                "status": "SUCESSO", 
                "total_sinais": total_sinais,
                "tabelas": tabelas,
                "categorias": categorias
            }
        except Exception as e:
            self.log(f"❌ Erro na conexão com banco: {e}", "ERRO")
            return {"status": "ERRO", "erro": str(e)}
    
    def testar_dinamica_simbolica(self, limite=10):
        """Testa a aplicação de dinâmica simbólica em vários sinais"""
        self.log("🔍 Testando dinâmica simbólica...")
        resultados = []
        estatisticas = {"sucessos": 0, "falhas": 0, "entropias": []}
        
        try:
            conexao = self.obter_conexao_db()
            cursor = conexao.cursor()
            
            # Testar sinais de ambas as categorias
            for categoria in ['S', 'N']:
                cursor.execute("""
                    SELECT s.id, s.nome 
                    FROM sinais s
                    JOIN usuarios u ON s.idusuario = u.id
                    WHERE u.possui = %s
                    ORDER BY s.id 
                    LIMIT %s
                """, (categoria, limite//2))
                sinais_categoria = cursor.fetchall()
                
                self.log(f"  Testando {len(sinais_categoria)} sinais categoria '{categoria}'...")
                
                for sinal_id, nome in sinais_categoria:
                    try:
                        resultado = aplicar_dinamica_simbolica(sinal_id, m=3)
                        
                        if resultado and len(resultado.get('sequencia_binaria', [])) > 0:
                            entropia = resultado.get('entropia', 0)
                            estatisticas["entropias"].append(entropia)
                            estatisticas["sucessos"] += 1
                            
                            resultados.append({
                                "sinal_id": sinal_id,
                                "nome": nome,
                                "categoria": categoria,
                                "status": "SUCESSO",
                                "entropia": entropia,
                                "limiar": resultado.get('limiar', 0),
                                "sequencia_length": len(resultado.get('sequencia_binaria', []))
                            })
                        else:
                            estatisticas["falhas"] += 1
                            resultados.append({
                                "sinal_id": sinal_id,
                                "nome": nome,
                                "categoria": categoria,
                                "status": "FALHA",
                                "erro": "Sequência vazia"
                            })
                            
                    except Exception as e:
                        estatisticas["falhas"] += 1
                        resultados.append({
                            "sinal_id": sinal_id,
                            "nome": nome,
                            "categoria": categoria,
                            "status": "ERRO",
                            "erro": str(e)
                        })
            
            cursor.close()
            conexao.close()
            
            # Estatísticas
            total_testes = estatisticas["sucessos"] + estatisticas["falhas"]
            taxa_sucesso = (estatisticas["sucessos"] / total_testes * 100) if total_testes > 0 else 0
            
            if estatisticas["entropias"]:
                entropia_media = np.mean(estatisticas["entropias"])
                entropia_min = np.min(estatisticas["entropias"])
                entropia_max = np.max(estatisticas["entropias"])
                
                self.log(f"✅ Dinâmica simbólica: {estatisticas['sucessos']}/{total_testes} sinais processados ({taxa_sucesso:.1f}%)")
                self.log(f"   Entropia média: {entropia_media:.4f}")
                self.log(f"   Entropia min/max: {entropia_min:.4f} / {entropia_max:.4f}")
            else:
                self.log(f"❌ Dinâmica simbólica: Nenhum sinal processado com sucesso")
            
        except Exception as e:
            self.log(f"❌ Erro geral na dinâmica simbólica: {e}", "ERRO")
            
        return {
            "status": "SUCESSO" if estatisticas["sucessos"] > 0 else "ERRO",
            "resultados": resultados,
            "estatisticas": estatisticas
        }
    
    def testar_classificador(self):
        """Testa o classificador de machine learning"""
        self.log("🔍 Testando classificador de ML...")
        
        try:
            classifier = EEGClassifier()
            
            # Teste 1: Carregar modelo existente
            self.log("  Testando carregamento de modelo...")
            try:
                classifier.carregar_modelo()
                if classifier.is_trained:
                    self.log("    ✅ Modelo carregado com sucesso")
                    modelo_status = "CARREGADO"
                else:
                    self.log("    ⚠️ Modelo não estava treinado")
                    modelo_status = "NAO_TREINADO"
            except Exception as e:
                self.log(f"    ❌ Erro ao carregar modelo: {e}")
                modelo_status = "ERRO_CARREGAMENTO"
            
            # Teste 2: Criar dataset
            self.log("  Testando criação de dataset...")
            try:
                X, y, nomes = classifier.criar_dataset(limite=20)
                self.log(f"    ✅ Dataset criado: {X.shape[0]} amostras, {X.shape[1]} features")
                
                # Estatísticas do dataset
                if len(y) > 0:
                    classe_sim = np.sum(y == 1)
                    classe_nao = np.sum(y == 0)
                    self.log(f"    Distribuição: {classe_sim} 'Sim' / {classe_nao} 'Não'")
                
                dataset_status = "SUCESSO"
            except Exception as e:
                self.log(f"    ❌ Erro ao criar dataset: {e}")
                dataset_status = "ERRO"
                X, y, nomes = None, None, None
            
            # Teste 3: Treinar modelo
            if X is not None and y is not None and len(y) > 0:
                self.log("  Testando treinamento de modelo...")
                try:
                    classifier.criar_modelo(tipo_modelo='random_forest')
                    resultado_treino = classifier.treinar_modelo(X, y)
                    self.log("    ✅ Modelo treinado com sucesso")
                    treino_status = "SUCESSO"
                except Exception as e:
                    self.log(f"    ❌ Erro no treinamento: {e}")
                    treino_status = "ERRO"
            else:
                self.log("    ⚠️ Pulando treinamento - dataset insuficiente")
                treino_status = "PULADO"
            
            # Teste 4: Predição
            if classifier.is_trained:
                self.log("  Testando predições...")
                try:
                    conexao = self.obter_conexao_db()
                    cursor = conexao.cursor()
                    cursor.execute("SELECT id FROM sinais ORDER BY id LIMIT 5")
                    sinais_teste = [row[0] for row in cursor.fetchall()]
                    cursor.close()
                    conexao.close()
                    
                    predicoes = []
                    for sinal_id in sinais_teste:
                        try:
                            predicao = classifier.prever_sinal(sinal_id)
                            if predicao:
                                predicoes.append({
                                    "sinal_id": sinal_id, 
                                    "predicao": predicao['classe_predita'],
                                    "probabilidade": predicao['probabilidade']
                                })
                        except Exception as e:
                            predicoes.append({"sinal_id": sinal_id, "erro": str(e)})
                    
                    self.log(f"    ✅ {len(predicoes)} predições testadas")
                    predicao_status = "SUCESSO"
                except Exception as e:
                    self.log(f"    ❌ Erro nas predições: {e}")
                    predicao_status = "ERRO"
            else:
                predicao_status = "PULADO"
            
            return {
                "status": "SUCESSO",
                "modelo": modelo_status,
                "dataset": dataset_status,
                "treino": treino_status,
                "predicao": predicao_status
            }
            
        except Exception as e:
            self.log(f"❌ Erro geral no classificador: {e}", "ERRO")
            return {"status": "ERRO", "erro": str(e)}
    
    def testar_geracao_graficos(self):
        """Testa a geração de gráficos"""
        self.log("🔍 Testando geração de gráficos...")
        
        try:
            resultado = gerar_grafico_interativo(limite=5)
            
            if resultado and 'graficos_html' in resultado:
                sinais_processados = len(resultado['dados_sinais'])
                self.log(f"    ✅ Gráficos gerados: {sinais_processados} sinais")
                
                # Verificar categorias dos gráficos
                categorias = {}
                for sinal in resultado['dados_sinais']:
                    cat = sinal.get('possui', '?')
                    categorias[cat] = categorias.get(cat, 0) + 1
                
                for cat, total in categorias.items():
                    self.log(f"    Categoria '{cat}': {total} gráficos")
                
                return {
                    "status": "SUCESSO", 
                    "sinais_processados": sinais_processados,
                    "categorias": categorias
                }
            else:
                self.log("    ❌ Erro na geração de gráficos")
                return {"status": "ERRO", "erro": "Resultado vazio"}
                
        except Exception as e:
            self.log(f"❌ Erro na geração de gráficos: {e}", "ERRO")
            return {"status": "ERRO", "erro": str(e)}
    
    def testar_arquivos_estaticos(self):
        """Testa se os arquivos estáticos estão acessíveis"""
        self.log("🔍 Testando arquivos estáticos...")
        
        arquivos_necessarios = [
            "static/style.css",
            "templates/grafico.html",
            "templates/dashboard.html",
            "templates/testes.html",
            "templates/erro.html"
        ]
        
        resultados = []
        for arquivo in arquivos_necessarios:
            if os.path.exists(arquivo):
                resultados.append({"arquivo": arquivo, "status": "EXISTE"})
                self.log(f"    ✅ {arquivo}")
            else:
                resultados.append({"arquivo": arquivo, "status": "FALTANDO"})
                self.log(f"    ❌ {arquivo}")
        
        arquivos_ok = sum(1 for r in resultados if r["status"] == "EXISTE")
        self.log(f"✅ Arquivos estáticos: {arquivos_ok}/{len(arquivos_necessarios)} encontrados")
        
        return {"status": "SUCESSO", "resultados": resultados}
    
    def testar_dados_sinais(self):
        """Testa a qualidade dos dados dos sinais"""
        self.log("🔍 Testando qualidade dos dados...")
        
        try:
            conexao = self.obter_conexao_db()
            cursor = conexao.cursor()
            
            # Verificar sinais com valores
            cursor.execute("""
                SELECT s.id, s.nome, u.possui, COUNT(v.valor) as num_valores
                FROM sinais s
                JOIN usuarios u ON s.idusuario = u.id
                LEFT JOIN valores_sinais v ON s.id = v.idsinal
                GROUP BY s.id, s.nome, u.possui
                ORDER BY s.id
            """)
            
            sinais = cursor.fetchall()
            cursor.close()
            conexao.close()
            
            sinais_com_dados = 0
            sinais_sem_dados = 0
            total_valores = 0
            
            for sinal_id, nome, categoria, num_valores in sinais:
                if num_valores > 0:
                    sinais_com_dados += 1
                    total_valores += num_valores
                else:
                    sinais_sem_dados += 1
            
            self.log(f"   Total de sinais: {len(sinais)}")
            self.log(f"   Sinais com dados: {sinais_com_dados}")
            self.log(f"   Sinais sem dados: {sinais_sem_dados}")
            self.log(f"   Total de valores: {total_valores}")
            
            if sinais_com_dados > 0:
                media_valores = total_valores / sinais_com_dados
                self.log(f"   Média de valores por sinal: {media_valores:.1f}")
            
            return {
                "status": "SUCESSO",
                "total_sinais": len(sinais),
                "sinais_com_dados": sinais_com_dados,
                "sinais_sem_dados": sinais_sem_dados,
                "total_valores": total_valores
            }
            
        except Exception as e:
            self.log(f"❌ Erro ao testar dados: {e}", "ERRO")
            return {"status": "ERRO", "erro": str(e)}
    
    def executar_todos_testes(self):
        """Executa todos os testes do sistema"""
        self.log("🚀 Iniciando bateria completa de testes...")
        inicio = time.time()
        
        # Executar todos os testes
        self.resultados = {
            "banco": self.testar_conexao_banco(),
            "dados": self.testar_dados_sinais(),
            "dinamica_simbolica": self.testar_dinamica_simbolica(),
            "classificador": self.testar_classificador(),
            "graficos": self.testar_geracao_graficos(),
            "arquivos": self.testar_arquivos_estaticos()
        }
        
        # Calcular estatísticas
        total_testes = len(self.resultados)
        testes_sucesso = sum(1 for r in self.resultados.values() if r.get("status") == "SUCESSO")
        
        tempo_total = time.time() - inicio
        
        self.log(f"🎯 Testes concluídos em {tempo_total:.2f}s")
        self.log(f"📊 Resultado: {testes_sucesso}/{total_testes} testes passaram")
        
        return {
            "tempo_total": tempo_total,
            "total_testes": total_testes,
            "testes_sucesso": testes_sucesso,
            "resultados": self.resultados,
            "logs": self.logs
        }

# Função para uso direto
def executar_testes():
    """Função para executar testes diretamente"""
    testador = TestadorSistema()
    return testador.executar_todos_testes()

if __name__ == "__main__":
    resultado = executar_testes()
    print("\n" + "="*50)
    print("RESUMO DOS TESTES:")
    print(f"Tempo total: {resultado['tempo_total']:.2f}s")
    print(f"Testes passaram: {resultado['testes_sucesso']}/{resultado['total_testes']}")
    print("="*50)
