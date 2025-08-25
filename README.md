# 🧠 EEG Visualizer Pro

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13%2B-blue.svg)](https://postgresql.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Sistema avançado de análise e visualização de sinais EEG com Dinâmica Simbólica**

Um projeto de pesquisa científica que combina processamento de sinais neurais, análise matemática avançada e visualização web interativa para estudar padrões em eletroencefalogramas (EEG).

## 🚀 Funcionalidades

- **📊 Visualização Interativa**: Gráficos responsivos com Plotly.js
- **🧮 Dinâmica Simbólica**: Análise de padrões temporais em sinais EEG  
- **📈 Análise Estatística**: Histogramas de frequência e sequências binárias
- **🔍 Filtros Inteligentes**: Categorização por grupos (Sim/Não)
- **🗄️ Banco Robusto**: Armazenamento otimizado em PostgreSQL
- **📱 Design Responsivo**: Interface moderna e mobile-friendly
- **🤖 Machine Learning**: Classificação automática com redes neurais
- **📊 Entropia de Shannon**: Medida de complexidade e informação

## 🛠️ Tecnologias

| Área | Tecnologia |
|------|------------|
| **Backend** | Python, Flask, NumPy, Matplotlib |
| **Frontend** | HTML5, CSS3, JavaScript, Plotly.js |
| **Banco de Dados** | PostgreSQL, psycopg2 |
| **Análise** | Dinâmica Simbólica, Processamento de Sinais |
| **Machine Learning** | Scikit-learn, MLPClassifier, RandomForest |

## 📋 Pré-requisitos

- Python 3.8 ou superior
- PostgreSQL 13+ 
- pip (gerenciador de pacotes Python)

## ⚡ Instalação Rápida

1. **Clone o repositório**
```bash
git clone https://github.com/seu-usuario/eeg-visualizer-pro.git
cd eeg-visualizer-pro
```

2. **Instale as dependências**
```bash
pip install flask plotly psycopg2-binary numpy matplotlib scikit-learn pandas seaborn
```

3. **Configure o banco PostgreSQL**
```bash
# Crie o banco de dados
createdb eeg-projeto

# Execute o script de criação das tabelas
python data_base.py
```

4. **Carregue os dados EEG no banco de dados**
```bash
# Execute o script que carrega todos os arquivos .txt da pasta "Sinais EEG"
python modulo_funcoes.py
```

**⚠️ IMPORTANTE**: Este passo é **OBRIGATÓRIO** antes de usar a aplicação. O script irá:
- Carregar 40 arquivos da categoria "SIM" (pasta `Sinais EEG/sim/`)
- Carregar 40 arquivos da categoria "NÃO" (pasta `Sinais EEG/nao/`)
- Total: 80 sinais EEG inseridos no banco PostgreSQL

**Saída esperada:**
```
[OK] Inserido: s001.txt - Categoria: SIM
[OK] Inserido: s002.txt - Categoria: SIM
...
[OK] Inserido: a001.txt - Categoria: NAO
[OK] Inserido: a002.txt - Categoria: NAO
...
```

**Se o banco estiver vazio**, você verá este erro ao iniciar a aplicação:
```
Categoria 'S': 0 sinais
Categoria 'N': 0 sinais
❌ Erro durante treinamento: Nenhuma feature foi extraída com sucesso!
```

5. **Inicie o servidor**
```bash
python app.py
```

6. **Acesse a aplicação**
```
http://localhost:5000
```

## 📂 Estrutura do Projeto

```
eeg-visualizer-pro/
├── 📄 app.py                    # Servidor Flask principal
├── 🧮 dinamica_simbolica.py     # Algoritmos de análise simbólica
├── 🤖 ml_classifier.py          # Sistema de machine learning
├── 🔧 modulo_funcoes.py         # Funções de processamento
├── 🗄️ data_base.py             # Configuração do banco
├── 📁 Sinais EEG/              # Dados brutos dos sinais
│   ├── sim/                    # Categoria "Sim" 
│   └── nao/                    # Categoria "Não"
├── 🎨 static/                  # Arquivos estáticos
│   ├── style.css              # Estilos CSS
│   └── *.png                  # Gráficos gerados
├── 📋 templates/               # Templates HTML
│   └── grafico.html           # Interface principal
└── 📚 README.md               # Este arquivo
```

## 🧪 Metodologia Científica

### Dinâmica Simbólica
1. **Binarização**: Conversão do sinal analógico para sequência binária usando a média como limiar
2. **Janelamento**: Criação de janelas deslizantes de 3 bits
3. **Simbolização**: Conversão dos grupos binários para valores decimais (0-7)
4. **Análise Estatística**: Cálculo de frequências relativas dos padrões

### Métricas Calculadas
- **Limiar Adaptativo**: Média aritmética do sinal
- **Distribuição de Padrões**: Histograma de frequência dos símbolos
- **Sequência Temporal**: Visualização da evolução binária

## 📊 Entropia de Shannon

### Conceito Teórico
A **Entropia de Shannon** é uma medida fundamental da teoria da informação que quantifica a incerteza ou complexidade de um sistema. No contexto de sinais EEG, ela mede a diversidade e imprevisibilidade dos padrões temporais.

### Fórmula Matemática
```
H(X) = -∑(p(x) × log(p(x)))
```

Onde:
- `H(X)` = Entropia de Shannon
- `p(x)` = Probabilidade do símbolo x
- `log` = Logaritmo natural (ln)

### Implementação no Projeto

```python
def calcular_entropia_shannon(frequencias):
    """
    Calcula a entropia de Shannon normalizada
    - Usa logaritmo natural (ln)
    - Retorna valor normalizado entre 0 e 1
    - Ignora valores de frequência relativa que sejam 0 ou 1
    """
    if not frequencias:
        return 0.0
    
    probabilidades = np.array(list(frequencias.values()))
    probabilidades_filtradas = probabilidades[(probabilidades > 0) & (probabilidades < 1)]
    
    if len(probabilidades_filtradas) == 0:
        return 0.0
        
    probabilidades_norm = probabilidades_filtradas / np.sum(probabilidades_filtradas)
    entropia_bruta = -np.sum(probabilidades_norm * np.log(probabilidades_norm))
    
    n_simbolos = len(probabilidades_filtradas)
    if n_simbolos > 1:
        entropia_maxima = np.log(n_simbolos)
        entropia_normalizada = entropia_bruta / entropia_maxima
    else:
        entropia_normalizada = 0.0
        
    return max(0.0, min(1.0, entropia_normalizada))
```

### Interpretação dos Valores
- **Entropia ≈ 0**: Sinal muito previsível, baixa complexidade
- **Entropia ≈ 1**: Sinal muito imprevisível, alta complexidade
- **Entropia ≈ 0.5**: Sinal com complexidade moderada

### Aplicação em EEG
- **Diagnóstico**: Sinais com entropia muito baixa podem indicar patologias
- **Classificação**: Diferentes estados cerebrais apresentam entropias distintas
- **Monitoramento**: Mudanças na entropia podem indicar alterações neurológicas

## 🤖 Rede Neural para Classificação

### Arquitetura da Rede
O sistema utiliza uma **Multi-Layer Perceptron (MLP)** implementada via scikit-learn:

```python
from sklearn.neural_network import MLPClassifier

# Configuração da rede neural
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Duas camadas ocultas
    activation='relu',             # Função de ativação ReLU
    solver='adam',                 # Otimizador Adam
    max_iter=1000,                 # Máximo de iterações
    random_state=42                # Semente para reprodutibilidade
)
```

### Features Utilizadas
O classificador extrai 15 features principais de cada sinal EEG:

#### Features de Entropia e Dinâmica Simbólica:
- **entropia_shannon**: Entropia normalizada dos padrões
- **total_padroes**: Número total de padrões únicos
- **padroes_unicos**: Quantidade de símbolos distintos
- **entropia_frequencias**: Entropia da distribuição de frequências

#### Features Estatísticas dos Valores Brutos:
- **media_valores**: Média aritmética do sinal
- **desvio_padrao**: Desvio padrão dos valores
- **variancia**: Variância dos dados
- **skewness**: Assimetria da distribuição
- **kurtosis**: Curtose (achatamento) da distribuição
- **amplitude**: Diferença entre máximo e mínimo
- **rms**: Root Mean Square (valor eficaz)

#### Features da Sequência Binária:
- **proporcao_uns**: Proporção de valores '1' na sequência
- **transicoes**: Número de transições 0→1 e 1→0
- **comprimento_sequencia**: Tamanho da sequência binária

### Processo de Treinamento

```python
# 1. Extração de features
features = classifier.extrair_features_sinal(id_sinal)

# 2. Criação do dataset
X, y = classifier.criar_dataset(limite=100)

# 3. Normalização dos dados
X_scaled = classifier.scaler.fit_transform(X)

# 4. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. Treinamento da rede
classifier.treinar_modelo(X_train, y_train)
```

### Métricas de Avaliação
- **Acurácia**: Proporção de classificações corretas
- **Precisão**: Proporção de verdadeiros positivos
- **Recall**: Sensibilidade do modelo
- **F1-Score**: Média harmônica entre precisão e recall
- **Matriz de Confusão**: Visualização dos erros de classificação

### Exemplo de Uso

```python
from ml_classifier import EEGClassifier

# Inicializa o classificador
classifier = EEGClassifier()

# Carrega modelo treinado
classifier.carregar_modelo('modelo_eeg.pkl')

# Faz predição em um novo sinal
resultado = classifier.prever_sinal(id_sinal=321)
print(f"Classe predita: {resultado['classe']}")
print(f"Probabilidade: {resultado['probabilidade']:.2f}")
```

## 📊 Exemplo de Uso

```python
from dinamica_simbolica import aplicar_dinamica_simbolica

# Analisa um sinal específico
resultado = aplicar_dinamica_simbolica(id_sinal=321, m=3)

print(f"Limiar: {resultado['limiar']:.2f}")
print(f"Entropia de Shannon: {resultado['entropia']:.4f}")
print(f"Padrões encontrados: {len(resultado['grupos_binarios'])}")
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Crie um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Autores

- **Eduardo Henrique** - *Desenvolvimento e Pesquisa*
- **Karollyne** - *Análise e Validação*

**Projeto PIBITI** - Programa Institucional de Bolsas de Iniciação em Desenvolvimento Tecnológico e Inovação

## 🔧 Troubleshooting

### Problemas Comuns

#### 1. **Banco de dados vazio**
**Sintoma:** Erro "Nenhuma feature foi extraída com sucesso!"
**Solução:** Execute `python modulo_funcoes.py` para carregar os dados

#### 2. **Erro de conexão com PostgreSQL**
**Sintoma:** `psycopg2.OperationalError: connection to server failed`
**Solução:** 
- Verifique se o PostgreSQL está rodando
- Confirme as credenciais em `data_base.py` e `modulo_funcoes.py`
- Crie o banco: `createdb eeg-projeto`

#### 3. **Erro de dependências**
**Sintoma:** `ModuleNotFoundError: No module named 'psycopg2'`
**Solução:** `pip install psycopg2-binary`

#### 4. **Modelo não treinado**
**Sintoma:** `FileNotFoundError: modelo_eeg.pkl`
**Solução:** Acesse `/testes` e clique em "Retreinar Modelo"

### Verificação Rápida do Sistema
```bash
# Teste completo do sistema (via interface web)
# Acesse http://localhost:5000/testes e clique em "Executar Testes"

# Ou via linha de comando
python testes_sistema.py
```

## 📞 Contato

- 📧 Email: [seu-email@exemplo.com](mailto:seu-email@exemplo.com)
- 🐱 GitHub: [@seu-usuario](https://github.com/seu-usuario)
- 📋 Issues: [Reportar problemas](https://github.com/seu-usuario/eeg-visualizer-pro/issues)

## 🏆 Agradecimentos

- Equipe do laboratório de neurociência
- Coordenação do programa PIBITI
- Comunidade Python científica

---

⭐ **Se este projeto foi útil, dê uma estrela no repositório!** 

---

## Como resolver

A solução mais simples e garantida é **fazer downgrade do NumPy para a versão 1.x**.  
Siga estes passos no terminal:

1. **Desinstale o NumPy atual:**
   ```
   pip uninstall numpy
   ```

2. **Instale uma versão compatível (exemplo: 1.26.4):**
   ```
   pip install numpy==1.26.4
   ```

3. **(Opcional, mas recomendado) Reinstale o Matplotlib para garantir compatibilidade:**
   ```
   pip install --force-reinstall matplotlib
   ```

4. **Tente rodar novamente:**
   ```
   python app.py
   ```

---

Se aparecer outro erro, envie aqui para eu te ajudar! 