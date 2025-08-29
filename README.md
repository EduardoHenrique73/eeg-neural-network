# Sistema de Análise EEG - PIBITI

Sistema completo para análise de sinais EEG usando dinâmica simbólica e machine learning.

## 🚀 Funcionalidades Principais

- **Análise de Sinais EEG** com dinâmica simbólica
- **Classificação automática** usando machine learning
- **Visualização interativa** de gráficos e histogramas
- **Dashboard** com estatísticas do sistema
- **Sistema de testes** automatizados
- **Upload de arquivos EEG** para análise individual
- **Logs em tempo real** de todos os processos

## 📋 Pré-requisitos

- Python 3.8+
- PostgreSQL
- Dependências listadas em `requirements.txt`

## 🛠️ Instalação

1. **Clone o repositório**
```bash
git clone <url-do-repositorio>
cd PIBITI
```

2. **Instale as dependências**
```bash
pip install -r requirements.txt
```

3. **Configure o banco de dados PostgreSQL**
```bash
# Crie um banco chamado 'eeg-projeto'
# Usuário: postgres
# Senha: EEG@321
```

4. **Carregue os dados EEG no banco de dados**
```bash
python modulo_funcoes.py
```
**Nota:** Se aparecer "Nenhuma feature foi extraída com sucesso!", é normal se o banco estiver vazio.

5. **Execute a aplicação**
```bash
python app.py
```

6. **Acesse a interface**
```
http://localhost:5000
```

## 🎯 Como Usar

### **Análise de Sinais Existentes**
1. Acesse a página principal
2. Use o filtro para selecionar "Grupo Sim" ou "Grupo Não"
3. Visualize os gráficos e análises

### **Upload de Arquivo EEG Individual**
1. Na página principal, clique em "Escolher Arquivo EEG"
2. Selecione um arquivo `.txt` com dados EEG (um valor por linha)
3. Clique em "Processar"
4. Aguarde o processamento automático:
   - ✅ Inserção no banco de dados
   - ✅ Cálculo de dinâmica simbólica
   - ✅ Geração de histogramas
   - ✅ Cálculo de entropia
   - ✅ Predição ML (se modelo treinado)
5. Visualize os resultados completos

### **Sistema de Testes**
1. Acesse a página de testes (`/testes`)
2. Clique em "Executar Testes" para verificar todo o sistema
3. Clique em "Retreinar Modelo" para atualizar o classificador ML
4. Acompanhe os logs em tempo real

### **Dashboard**
- Visualize estatísticas gerais do sistema
- Veja sinais recentes processados
- Monitore entropia média

## 🧪 Teste do Sistema de Upload

Para testar o sistema de upload:

```bash
# Criar arquivo de teste
python teste_upload.py

# Executar aplicação
python app.py

# Acessar e fazer upload do arquivo teste_eeg_upload.txt
```

## 📊 Estrutura do Projeto

```
PIBITI/
├── app.py                 # Aplicação Flask principal
├── ml_classifier.py       # Classificador de machine learning
├── dinamica_simbolica.py  # Análise de dinâmica simbólica
├── modulo_funcoes.py      # Funções auxiliares
├── testes_sistema.py      # Sistema de testes
├── teste_upload.py        # Script de teste para upload
├── templates/             # Templates HTML
├── static/               # Arquivos estáticos (CSS, imagens)
├── uploads/              # Pasta para arquivos enviados
└── Sinais EEG/           # Dados EEG originais
```

## 🔧 Troubleshooting

### **Banco de dados vazio**
```bash
python modulo_funcoes.py
```

### **Erro de conexão PostgreSQL**
- Verifique se o PostgreSQL está rodando
- Confirme as credenciais em `app.py`

### **Dependências faltando**
```bash
pip install -r requirements.txt
```

### **Modelo não treinado**
- Acesse `/testes` e clique em "Retreinar Modelo"
- Ou execute `python app.py` (treina automaticamente)

### **Erro no upload de arquivo**
- Verifique se o arquivo é `.txt`
- Confirme que contém apenas valores numéricos (um por linha)
- Verifique o tamanho do arquivo (máximo 16MB)

## 📈 Logs e Monitoramento

O sistema possui logs detalhados para:
- **Retreinamento do modelo ML**
- **Execução de testes do sistema**
- **Processamento de uploads**
- **Análise de dinâmica simbólica**

Acesse `/testes` para ver todos os logs em tempo real.

## 🤝 Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto é parte do PIBITI - Programa Institucional de Bolsas de Iniciação em Desenvolvimento Tecnológico e Inovação. 