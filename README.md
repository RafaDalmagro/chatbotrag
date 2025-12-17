# RAG para consulta de leis municipais de Caçador.

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)
![LangChain](https://img.shields.io/badge/langchain-%231C3C3C.svg?style=for-the-badge&logo=langchain&logoColor=white)
[![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-no-red.svg)](https://bitbucket.org/lbesson/ansi-colors)

Este projeto consiste em um sistema baseado em Inteligência Artificial capaz de responder perguntas relacionadas a algumas das leis do município de Caçador. A aplicação tem como objetivo facilitar o acesso à informação legislativa, permitindo que cidadãos tirem dúvidas de forma simples e rápida.

O sistema foi desenvolvido em Python, utilizando a biblioteca Docling para realizar o processamento e conversão de arquivos PDF para o formato Markdown. A lógica de funcionamento e a configuração do modelo de IA foram implementadas com o auxílio do LangChain, possibilitando a organização do fluxo de perguntas e respostas.

O projeto foi desenvolvido como atividade final da disciplina de Inteligência Artificial do curso de Sistemas de Informação. Seu principal propósito é aplicar, na prática, as técnicas estudadas ao longo da disciplina, ao mesmo tempo em que oferece uma solução útil para a comunidade local.

## Instalação

### Pré-requisitos

- **Python**: versão **3.11**
- **pip** instalado
- Conta e chave de API da **OpenAI**

### Passo a passo

1. **Clonar o repositório em sua máquina via terminal**

```bash
git clone https://github.com/RafaDalmagro/chatbotrag.git
cd chatbotrag
```

2. **(Recomendado) Criar e ativar um ambiente virtual**

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. **Instalar as dependências do projeto com o [pip](https://pip.pypa.io/en/stable/)**

```bash
pip install -r requirements.txt
```

4. **Configurar a variável de ambiente da OpenAI** 
- Crie um arquivo `.env` na raiz do projeto e defina sua `OPENAI_API_KEY`.  
- Caso não possua uma chave da OpenAI, será necessário ajustar a configuração do modelo utilizado no projeto.

```bash
OPENAI_API_KEY=seu_token_aqui
```

## Guia de uso

1. **Preparar o ambiente**
   - Certifique-se de ter o Python 3.11 instalado.
   - (Opcional, mas recomendado) Crie e ative um ambiente virtual.
   - Instale (ou garanta que já instalou) as dependências do projeto:

   ```bash
   pip install -r requirements.txt
   pip install docling langchain-core jupyter
   ```

2. **Configurar a chave da OpenAI**
   - Crie um arquivo `.env` na raiz do projeto com a sua chave de API (caso ainda não exista):

   ```bash
   OPENAI_API_KEY=seu_token_aqui
   ```

3. **Abrir e executar o notebook**
   - Abra o arquivo `chatbot.ipynb` em um ambiente de notebook (Jupyter, VS Code, Cursor, etc.).
   - Execute as células na ordem:
     - Importação e instalação de bibliotecas (se necessário).
     - Carregamento das leis em formato Markdown a partir da pasta `data/leis`.
     - Divisão dos textos em *chunks*.
     - Criação da base vetorial com o ChromaDB (pasta `chroma_db`).
     - Configuração da *chain* de perguntas e respostas (`qa_chain`).

4. **Fazer perguntas sobre as leis**
   - Com a `qa_chain` configurada, utilize a função de exemplo no final do notebook:

   ```python
   ask_question(qa_chain, "Qual é a sua pergunta sobre as leis de Caçador?")
   ```

   - O modelo responderá com base nas leis municipais carregadas (Lei Orgânica e demais arquivos presentes em `data/leis`), citando trechos relevantes quando possível.

## Licença

[MIT](https://choosealicense.com/licenses/mit/)
