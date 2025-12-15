import os
import sys
import time
import shutil

from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

print("DEBUG .env LOADED?", load_dotenv())
print("DEBUG KEY:", os.getenv("OPENAI_API_KEY"))

MARKDOWN_DIR = "data/leis"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

class LeiOrganicaChatbot:
    
    def __init__(self):
        load_dotenv()
        self.vectorstore = None
        self.qa_chain = None
        
    def initialize(self, force_rebuild=False):    
        if os.path.exists(CHROMA_DB_DIR) and not force_rebuild:
            self._load_vectorstore()
        else:
            self._build_vectorstore()
        
        self._setup_qa_chain()
        print("✅ Sistema inicializado!\n")
    
    def _load_vectorstore(self):
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
    
    def _build_vectorstore(self):

        md_dir = Path(MARKDOWN_DIR)
        md_files = sorted(md_dir.glob("*.md"))

        if not md_files:
            raise FileNotFoundError(f"Nenhum arquivo .md encontrado em: {MARKDOWN_DIR}")

        print(f"📄 Encontrados {len(md_files)} arquivos .md em {MARKDOWN_DIR}")

        documents = []
        
        for f in md_files:
            print(f"  - Carregando: {f.name}")
            loader = UnstructuredMarkdownLoader(str(f), mode="single")
            docs = loader.load()
            
            for d in docs:
                d.metadata = d.metadata or {}
                d.metadata["source"] = f.name
                d.metadata["source_path"] = str(f)
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
            separators=[
                "\nArt. ",
                "\nArtigo ",
                "\n§",
                "\nI -", "\nII -", "\nIII -",
                "\n\n",
                "\n",
                ". ",
                " "
            ]
        )

        chunks = text_splitter.split_documents(documents)
        print(f"✅ Total de chunks gerados: {len(chunks)}")

        if os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )

    def responder_fn_com_etapas(pergunta: str, retriever, llm, prompt_builder, k=5):
        t0 = time.perf_counter()
        docs = retriever.get_relevant_documents(pergunta)[:k]
        t1 = time.perf_counter()

        prompt = prompt_builder(pergunta, docs)
        t2 = time.perf_counter()

        resposta = llm.invoke(prompt)
        t3 = time.perf_counter()

        return {
            "answer": resposta,
            "timing": {
                "retrieval_s": t1 - t0,
                "prompt_build_s": t2 - t1,
                "generation_s": t3 - t2,
                "total_s": t3 - t0
            }
        }

    def _setup_qa_chain(self):

        template = """Você é um assistente especializado nas Leis do Município de Caçador/SC.
        Use o contexto abaixo para responder a pergunta.
        
        REGRAS:
        - Cite artigos específicos quando relevante (ex: "Conforme art. 23...")
        - Se não souber a resposta com base no contexto, diga que não sabe.
        - Utilize citações das leis quando possível. 
        - Utilize até 30 linhas para a resposta.
        - Seja claro e direto, respondendo de forma concisa.
        
        Contexto: {context}
        
        Pergunta: {question}
        
        Resposta:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def ask(self, question, show_sources=False):

        result = self.qa_chain({"query": question})
        
        answer = result['result']
        sources = result['source_documents']

        if show_sources:
            return answer, sources
        return answer
    
    def chat_loop(self):
        print("="*80)
        print("🤖 CHATBOT - LEIS DE CAÇADOR/SC")
        print("="*80)
        print("\n" + "-"*80 + "\n")
        
        show_sources = False
        
        while True:
            try:
                question = input("👤 Faça sua pergunta: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['sair', 'exit', 'quit']:
                    print("\n👋 Até logo!")
                    break
                
                if question.lower() == 'fontes':
                    show_sources = not show_sources
                    status = "ativado" if show_sources else "desativado"
                    print(f"📚 Mostrar fontes: {status}")
                    continue
                
                if question.lower() == 'limpar':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                
                print("\n🔍 Processando...\n")
                
                if show_sources:
                    answer, sources = self.ask(question, show_sources=True)
                    print(f"🤖 Resposta:\n{answer}")
                    print(f"\n📚 Baseado em {len(sources)} trechos da lei")
                    show_sources = False
                else:
                    answer = self.ask(question)
                    print(f"🤖 Resposta:\n{answer}")
                
                print("\n" + "-"*80 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Até logo!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}\n")

def main():
    if not os.path.isdir(MARKDOWN_DIR):
        print(f"Pasta não encontrada: {MARKDOWN_DIR}")
        print("\nCertifique-se de ter a pasta com arquivos .md na localização correta.")
        sys.exit(1)

    from pathlib import Path
    if not any(Path(MARKDOWN_DIR).glob("*.md")):
        print(f"Nenhum arquivo .md encontrado em: {MARKDOWN_DIR}")
        sys.exit(1)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY não encontrada no .env")
        sys.exit(1)
    
    chatbot = LeiOrganicaChatbot()

    force_rebuild = "--rebuild" in sys.argv
    chatbot.initialize(force_rebuild=force_rebuild)

    chatbot.chat_loop()


if __name__ == "__main__":
    main()
