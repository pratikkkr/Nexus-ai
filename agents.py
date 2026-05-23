from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
from tools import web_search, scrape_url
import os

# =========================
# ENV
# =========================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

# =========================
# MODEL (GROQ)
# =========================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# =========================
# SEARCH AGENT
# =========================

def build_search_agent():
    return create_react_agent(
        model=llm,
        tools=[web_search]
    )

# =========================
# READER AGENT
# =========================

def build_reader_agent():
    return create_react_agent(
        model=llm,
        tools=[scrape_url]
    )

# =========================
# WRITER CHAIN
# =========================

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write structured reports."),
    ("human", """Topic: {topic}

Research:
{research}

Write:
- Introduction
- Key Findings (3+ points)
- Conclusion
- Sources"""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# =========================
# CRITIC CHAIN
# =========================

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict research critic."),
    ("human", """Report:
{report}

Give:
Score: X/10
Strengths
Weaknesses
Verdict"""),
])

critic_chain = critic_prompt | llm | StrOutputParser()