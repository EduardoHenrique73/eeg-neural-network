# 🧪 Sistema de Testes - EEG Visualizer Pro

Este documento explica como usar o novo sistema de testes implementado no projeto EEG.

## 📋 Funcionalidades Implementadas

### 1. **Módulo de Testes Automatizados** (`testes_sistema.py`)
- Testa conexão com banco de dados
- Testa aplicação de dinâmica simbólica
- Testa classificador de machine learning
- Testa geração de gráficos
- Testa arquivos estáticos

### 2. **Interface Web para Testes** (`/testes`)
- Página dedicada para executar testes
- Visualização em tempo real dos logs
- Status de execução com indicadores visuais
- Resumo dos resultados dos testes

### 3. **Sistema de Logs em Tempo Real**
- Logs detalhados durante retreinamento
- Logs detalhados durante execução de testes
- Interface web atualizada automaticamente
- Histórico de execuções

### 4. **Retreinamento com Logs**
- Retreinamento em background
- Logs em tempo real do processo
- Status visual do progresso
- Não bloqueia a interface

## 🚀 Como Usar

### **1. Acessar a Página de Testes**
```
http://localhost:5000/testes
```

### **2. Executar Testes**
1. Clique no botão **"🧪 Executar Testes"**
2. Acompanhe os logs em tempo real
3. Veja o resumo dos resultados ao final

### **3. Retreinar Modelo**
1. Clique no botão **"🔄 Retreinar Modelo"**
2. Acompanhe o progresso nos logs
3. Aguarde a conclusão

### **4. Navegação**
- **🏠 Início**: Página principal com análise de sinais
- **📊 Dashboard**: Estatísticas gerais do sistema
- **🧪 Testes**: Página de testes (nova funcionalidade)

## 📊 Tipos de Testes Executados

### **1. Teste de Conexão com Banco**
- Verifica se o PostgreSQL está acessível
- Conta total de sinais no banco
- Testa queries básicas

### **2. Teste de Dinâmica Simbólica**
- Aplica dinâmica simbólica em 5 sinais de teste
- Verifica cálculo de entropia
- Testa geração de sequências binárias

### **3. Teste do Classificador ML**
- Testa carregamento de modelo existente
- Cria dataset de treinamento
- Treina novo modelo se necessário
- Testa predições em sinais

### **4. Teste de Geração de Gráficos**
- Testa criação de gráficos interativos
- Verifica processamento de sinais
- Testa integração com Plotly

### **5. Teste de Arquivos Estáticos**
- Verifica existência de templates HTML
- Testa arquivos CSS
- Valida estrutura de diretórios

## 🔧 Execução Manual de Testes

### **Via Linha de Comando**
```bash
# Executar testes do sistema
python testes_sistema.py

# Executar teste completo
python teste_sistema_completo.py
```

### **Via Python Interativo**
```python
from testes_sistema import TestadorSistema

# Criar instância do testador
testador = TestadorSistema()

# Executar todos os testes
resultado = testador.executar_todos_testes()

# Ver resultados
print(f"Testes passaram: {resultado['testes_sucesso']}/{resultado['total_testes']}")
print(f"Tempo total: {resultado['tempo_total']:.2f}s")
```

## 📈 Interpretação dos Resultados

### **Status dos Testes**
- **✅ SUCESSO**: Teste passou completamente
- **❌ ERRO**: Erro crítico no teste
- **⚠️ FALHA**: Teste falhou mas não é crítico

### **Indicadores Visuais**
- **⏸️ Ocioso**: Nenhum teste em execução
- **🔄 Executando**: Teste em andamento
- **✅ Concluído**: Teste finalizado com sucesso
- **❌ Erro**: Teste falhou

### **Logs de Execução**
- Timestamp em cada log
- Tipo de mensagem (INFO, ERRO, etc.)
- Detalhes específicos de cada etapa
- Emojis para facilitar identificação

## 🛠️ Estrutura dos Arquivos

```
PIBITI/
├── app.py                          # Aplicação Flask principal
├── testes_sistema.py               # Módulo de testes
├── teste_sistema_completo.py       # Script de teste completo
├── templates/
│   ├── testes.html                 # Página de testes
│   ├── grafico.html                # Página principal (atualizada)
│   └── dashboard.html              # Dashboard (atualizado)
└── README_TESTES.md                # Esta documentação
```

## 🔍 Troubleshooting

### **Problemas Comuns**

1. **Erro de Conexão com Banco**
   - Verifique se o PostgreSQL está rodando
   - Confirme credenciais no código
   - Teste conexão manual

2. **Erro nos Testes de ML**
   - Verifique se há dados suficientes no banco
   - Confirme se o modelo pode ser treinado
   - Verifique dependências Python

3. **Interface não Atualiza**
   - Recarregue a página
   - Verifique console do navegador
   - Confirme se o Flask está rodando

### **Logs de Debug**
- Todos os logs são salvos em tempo real
- Verifique o console do servidor Flask
- Use o console do navegador para erros JavaScript

## 📝 Exemplo de Uso Completo

1. **Iniciar o Sistema**
   ```bash
   python app.py
   ```

2. **Acessar Interface**
   ```
   http://localhost:5000
   ```

3. **Ir para Testes**
   - Clique em "🧪 Testes" no menu
   - Ou acesse diretamente: `http://localhost:5000/testes`

4. **Executar Testes**
   - Clique em "🧪 Executar Testes"
   - Acompanhe os logs em tempo real
   - Veja o resumo final

5. **Retreinar Modelo**
   - Clique em "🔄 Retreinar Modelo"
   - Acompanhe o progresso
   - Aguarde conclusão

## 🎯 Benefícios do Sistema

- **Transparência**: Logs detalhados de todas as operações
- **Confiabilidade**: Testes automatizados garantem funcionamento
- **Facilidade**: Interface web intuitiva
- **Monitoramento**: Status em tempo real
- **Debugging**: Identificação rápida de problemas

## 🔄 Atualizações Futuras

- [ ] Testes de performance
- [ ] Testes de integração
- [ ] Relatórios detalhados
- [ ] Exportação de resultados
- [ ] Testes automatizados por agendamento

---

**Desenvolvido para o projeto PIBITI - Sistema de Análise de Sinais EEG**
