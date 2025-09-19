# 🧠 AI-Powered Citation Enhancer

An AI agent built with **LangGraph** that takes user-written sentences or paragraphs, generates a search query, retrieves a relevant source using LangSearch, and rewrites the input with integrated quotes and APA-style citations — all in a formal, essay-like tone.

---

## 🚀 What It Does

- Accepts a sentence or paragraph as input.
- Uses an LLM (via Azure OpenAI) to generate a search query.
- Retrieves a credible source using the **LangSearch API**.
- Extracts a quote or snippet from the source.
- Rewrites the original input to include the quote and an APA-style in-text citation.
- Outputs a polished, citation-enhanced version of the input.

---

## 🛠️ Tools Used

- **LangGraph** – Orchestrates the agent workflow as a stateful graph.
- **LangChain** – Powers prompt execution and LLM interactions.
- **Azure OpenAI (GPT-4o)** – Handles query generation and rewriting.
- **LangSearch API** – Retrieves real-time web sources and snippets.
- **Python** – Core implementation.
- **dotenv** – Manages API keys and environment variables.

---

## 📦 Installation

```bash
git clone https://github.com/your-username/citation-ai.git
cd citation-ai
pip install -r requirements.txt
```

Create a `.env` file with the following with the content from `.env.example`

---

## 📄 Output Format

The rewritten output includes:
- A formal version of the input sentence or paragraph.
- At least one quote from a credible source.
- An APA-style in-text citation.
- A “Sources” section with full APA-style references.

---

## 💡 Use Cases

- Academic writing with embedded citations.
- Journalistic content backed by real-time sources.
- Blog posts or opinion pieces with supporting evidence.
- Research assistants for students or analysts.

---

## 📄 License

MIT License
