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

## 🛠️ Tecnologias

| Área | Tecnologia |
|------|------------|
| **Backend** | Python, Flask, NumPy, Matplotlib |
| **Frontend** | HTML5, CSS3, JavaScript, Plotly.js |
| **Banco de Dados** | PostgreSQL, psycopg2 |
| **Análise** | Dinâmica Simbólica, Processamento de Sinais |

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
pip install flask plotly psycopg2-binary numpy matplotlib
```

3. **Configure o banco PostgreSQL**
```bash
# Crie o banco de dados
createdb eeg-projeto

# Execute o script de criação das tabelas
python data_base.py
```

4. **Carregue os dados EEG**
```bash
python -c "from modulo_funcoes import processar_arquivos; processar_arquivos()"
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

## 📊 Exemplo de Uso

```python
from dinamica_simbolica import aplicar_dinamica_simbolica

# Analisa um sinal específico
resultado = aplicar_dinamica_simbolica(id_sinal=321, m=3)

print(f"Limiar: {resultado['limiar']:.2f}")
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