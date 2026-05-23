# ⚡ Nexus AI — Autonomous Multi-Agent Research Pipeline

> **4 specialized AI agents** work in sequence to search the web, read sources, write structured reports, and audit quality — all from a single research prompt.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange?style=flat-square)](https://langchain.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-red?style=flat-square)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

---

## 📸 Demo

> *Enter any research topic → watch 4 agents execute in real-time → get a structured, cited intelligence report with a quality score.*

![Nexus AI Pipeline](https://placehold.co/900x420/080c14/a78bfa?text=Nexus+AI+%E2%80%94+Live+Demo+Screenshot)

---

## 🧠 How It Works

The pipeline is a **4-stage agentic workflow** using LangGraph ReAct agents and LangChain LCEL chains:

```
User Prompt
    │
    ▼
┌─────────────────────┐
│  🔍 Search Agent    │  ← ReAct Agent + Tavily API → 5 sources with titles, URLs, snippets
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  📖 Reader Agent    │  ← ReAct Agent + BeautifulSoup → scrapes top URL, extracts full content
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  ✍️  Writer Chain   │  ← LCEL Chain → structured report: intro, findings, conclusion, sources
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  🎯 Critic Chain    │  ← LCEL Chain → score /10, strengths, weaknesses, verdict
└─────────────────────┘
          │
          ▼
     Final Report
```

---

## ✨ Features

- **Autonomous multi-agent orchestration** via LangGraph `create_react_agent`
- **Live pipeline status UI** — each agent shows Standby → Running → Done in real-time
- **Structured research output** with introduction, key findings, conclusion, and sources
- **Quality audit** — critic agent scores the report `/10` with strengths/weaknesses
- **Source explorer tab** — all scraped URLs displayed with titles and snippets
- **One-click presets** — 8 built-in research topics for quick demos
- **Markdown download** — export any report as a `.md` file
- **Groq LLaMA 3.3 70B** — fast, free-tier inference for all agent reasoning

---

## 🗂 Project Structure

```
nexus-ai/
│
├── app.py              # Streamlit UI — pipeline visualizer, tabs, CSS theming
├── agents.py           # Search agent, Reader agent, Writer chain, Critic chain
├── pipeline.py         # CLI entry point — runs the full pipeline end-to-end
├── tools.py            # LangChain tools — web_search (Tavily), scrape_url (BS4)
├── requirements.txt    # All dependencies
├── .env                # API keys (never commit — see .gitignore)
└── .gitignore
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — LLaMA 3.3 70B Versatile |
| Agent Framework | LangGraph `create_react_agent` |
| Chain Framework | LangChain LCEL — `ChatPromptTemplate \| LLM \| StrOutputParser` |
| Web Search | Tavily Search API |
| Web Scraping | BeautifulSoup4 + Requests |
| UI | Streamlit (custom CSS, dark theme) |
| Env Management | python-dotenv |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/nexus-ai.git
cd nexus-ai
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get your keys for free:
- **Groq:** https://console.groq.com
- **Tavily:** https://app.tavily.com

### 5. Run the app

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**CLI mode:**
```bash
python pipeline.py
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Groq API key for LLaMA inference |
| `TAVILY_API_KEY` | ✅ Yes | Tavily API key for web search |

---

## 🧩 Key Design Decisions

**Why LangGraph ReAct agents for Search and Reader?**
These steps require tool-calling in a reasoning loop — the agent decides *which* URL to scrape based on search context. ReAct handles this dynamically vs a hardcoded chain.

**Why LCEL chains for Writer and Critic?**
Writing and critiquing are single-pass, deterministic tasks — no tool-calling needed. LCEL chains are simpler, faster, and easier to prompt-engineer for structured output.

**Why Groq?**
LLaMA 3.3 70B on Groq runs at ~500 tokens/sec — the full pipeline completes in ~60 seconds, making it demo-friendly without OpenAI costs.

---

## 🔧 Customization

**Swap the LLM** — edit `agents.py`:
```python
# Use OpenAI instead of Groq
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

**Add more tools** — extend `tools.py` with any `@tool`-decorated function and pass it to the relevant agent in `agents.py`.

**Change search depth** — edit `tools.py`:
```python
results = tavily.search(query=query, max_results=10)  # default: 5
```

---

## 📋 Example Output

**Topic:** *"Latest breakthroughs in fusion energy 2025"*

```
## Introduction
Fusion energy reached a historic milestone in 2025 as three independent...

## Key Findings
1. NIF achieved net energy gain for the third consecutive quarter...
2. Commonwealth Fusion's SPARC magnet demonstrated 20T field strength...
3. EU's ITER project completed its central solenoid installation...

## Conclusion
2025 marks the year fusion transitioned from theoretical promise to...

## Sources
- nature.com/fusion-breakthrough-2025
- science.org/iter-update
...

---
Score: 8/10
Strengths: Well-structured, cited, technically accurate...
Weaknesses: Could include more quantitative data...
```

---

## 🗺 Roadmap

- [ ] Persistent research history with session storage
- [ ] Multi-URL parallel scraping for richer context
- [ ] Memory across sessions using FAISS vector store
- [ ] Export to PDF / DOCX
- [ ] Agent streaming output (token-by-token display)
- [ ] Add a Planner agent to decompose complex queries

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

**Arjun** — AI/ML Engineer  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

> Built as part of a hands-on agentic AI learning series.  
> Feedback and PRs welcome!
