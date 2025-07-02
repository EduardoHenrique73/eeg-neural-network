
# 🧠 Visualização Interativa de Sinais EEG

Este projeto tem como objetivo **processar sinais EEG em formato `.txt`**, armazená-los em um banco PostgreSQL, e exibi-los de forma **interativa e responsiva** usando `Flask` + `Plotly`.

---

## 📁 Estrutura do Projeto

```
PIBITI/
├── app.py                  # Servidor Flask
├── data_base.py            # Criação do banco de dados (PostgreSQL)
├── modulo_funcoes.py       # Processamento e geração dos gráficos
├── Sinais EEG/
│   ├── sim/                # Arquivos .txt da categoria "Sim"
│   └── nao/                # Arquivos .txt da categoria "Não"
├── static/
│   ├── style.css           # Estilos (opcional)
│   └── grafico_temp.html   # Arquivo gerado com os gráficos
└── templates/
    └── grafico.html        # Página HTML principal
```

---

## 🔧 Pré-requisitos

- Python 3.10 ou superior
- PostgreSQL instalado e configurado
- Pip atualizado

### 📦 Instale as dependências Python:

```bash
pip install flask plotly psycopg2-binary
```

---

## 🚀 Como rodar o projeto

### 1. Clone ou baixe o projeto

### 2. Crie o banco de dados PostgreSQL e configure as credenciais:

```bash
# Crie o banco no terminal
createdb eeg-projeto
```

> O projeto espera que o banco seja acessível com os seguintes dados:

- **Database**: `eeg-projeto`
- **Usuário**: `postgres`
- **Senha**: `EEG@321`
- **Host**: `localhost`
- **Porta**: `5432`

### 3. Crie a tabela no banco PostgreSQL:

```bash
python data_base.py
```

### 4. Adicione os arquivos `.txt` nas pastas `Sinais EEG/sim` e `Sinais EEG/nao`

### 5. Insira os dados no banco:

```bash
python -c "from modulo_funcoes import process_files; process_files()"
```

### 6. Inicie o servidor Flask:

```bash
python app.py
```

### 7. Acesse no navegador:

```
http://127.0.0.1:5000/
```

---

## 🧭 Funcionalidades

- ✅ Visualização interativa com rolagem horizontal
- ✅ Filtro por categoria: Sim (S), Não (N), ou Todos
- ✅ Botão para abrir gráfico em nova aba
- ✅ Design moderno com Plotly e responsividade leve

---

## 📌 Exemplo de arquivo `.txt`

Cada linha do arquivo deve conter **valores numéricos separados por espaço**, exemplo:

```
23.1 24.5 22.3 21.8 25.1 26.4 24.0 23.5
```

---

## 👨‍💻 Autor

**Desenvolvido por:** Eduardo Henrique e Karollyne / PIBITI  
Contato: [Seu e-mail ou LinkedIn opcional]

---
