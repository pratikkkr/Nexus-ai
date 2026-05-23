import sys
import importlib.machinery
from types import ModuleType

# ── Silence torchvision watcher/find_spec errors ──────────────────────────────
class _Dummy(ModuleType):
    def __getattr__(self, n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        return _Dummy(n)

for _n in ["torchvision","torchvision.transforms","torchvision.transforms.v2",
           "torchvision.transforms.v2.functional","torchvision.ops","torchvision.ops.boxes"]:
    _mod = _Dummy(_n)
    _mod.__spec__ = importlib.machinery.ModuleSpec(_n, loader=None)
    _mod.__spec__.submodule_search_locations = []
    sys.modules[_n] = _mod

import streamlit as st
import re
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nexus · AI Research",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    color: #c8ccd4;
    background: #080c14;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ══════════════════════════════════════════════════════════════════
   AMBIENT BACKGROUND
══════════════════════════════════════════════════════════════════ */
.bg-wrap {
    position: fixed; inset: 0; z-index: 0; overflow: hidden;
    pointer-events: none;
}
.bg-blob {
    position: absolute; border-radius: 50%;
    filter: blur(160px); opacity: 0.55;
    animation: floatBlob 14s ease-in-out infinite;
}
.bb1 {
    width: 800px; height: 800px;
    background: radial-gradient(circle, #4f1d96 0%, transparent 65%);
    top: -280px; left: -280px; animation-delay: 0s;
}
.bb2 {
    width: 600px; height: 600px;
    background: radial-gradient(circle, #0e4f6e 0%, transparent 65%);
    bottom: -200px; right: -200px; animation-delay: -6s;
}
.bb3 {
    width: 350px; height: 350px;
    background: radial-gradient(circle, #065f46 0%, transparent 65%);
    top: 45%; left: 55%; animation-delay: -11s;
}
@keyframes floatBlob {
    0%,100% { transform: translate(0,0) scale(1); }
    30%      { transform: translate(25px,-35px) scale(1.07); }
    60%      { transform: translate(-18px,22px) scale(0.94); }
}

/* Dot grid */
.bg-grid {
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image: radial-gradient(circle, rgba(255,255,255,0.055) 1px, transparent 1px);
    background-size: 28px 28px;
    mask-image: radial-gradient(ellipse 75% 70% at 50% 40%, black 20%, transparent 100%);
}

/* ══════════════════════════════════════════════════════════════════
   NAV
══════════════════════════════════════════════════════════════════ */
.nx-nav {
    position: sticky; top: 0; z-index: 200;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.9rem 3.5rem;
    background: rgba(8,12,20,0.8);
    backdrop-filter: blur(24px);
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.nx-logo {
    display: flex; align-items: center; gap: 0.75rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 800; color: #fff;
    letter-spacing: -0.02em;
}
.nx-logo-mark {
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, #7c3aed 0%, #0891b2 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; font-style: normal;
    box-shadow: 0 0 20px rgba(124,58,237,0.45), 0 0 0 1px rgba(255,255,255,0.08) inset;
}
.nx-nav-center { display: flex; gap: 0.3rem; }
.nx-nav-tab {
    font-size: 0.78rem; font-weight: 500; color: #475569;
    padding: 0.3rem 0.85rem; border-radius: 8px; cursor: default;
    border: 1px solid transparent;
    transition: all 0.2s;
}
.nx-nav-tab.on {
    color: #c4b5fd;
    background: rgba(124,58,237,0.1);
    border-color: rgba(124,58,237,0.2);
}
.nx-nav-live {
    display: flex; align-items: center; gap: 0.45rem;
    font-size: 0.72rem; font-weight: 500; color: #34d399;
}
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 8px #34d399;
    animation: livePulse 2.2s ease-in-out infinite;
}
@keyframes livePulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.35;transform:scale(0.75)} }

/* ══════════════════════════════════════════════════════════════════
   PAGE LAYOUT
══════════════════════════════════════════════════════════════════ */
.nx-page {
    max-width: 1060px; margin: 0 auto;
    padding: 0 2rem 8rem;
    position: relative; z-index: 1;
}

/* ══════════════════════════════════════════════════════════════════
   HERO
══════════════════════════════════════════════════════════════════ */
.nx-hero {
    padding: 5rem 0 1.5rem;
    text-align: center;
}

.nx-eyebrow {
    display: inline-flex; align-items: center; gap: 0.55rem;
    padding: 0.35rem 1.1rem;
    border: 1px solid rgba(124,58,237,0.28);
    border-radius: 100px;
    background: rgba(124,58,237,0.06);
    font-size: 0.67rem; font-weight: 700;
    letter-spacing: 0.22em; text-transform: uppercase;
    color: #a78bfa; margin-bottom: 2rem;
}
.eyebrow-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: #a78bfa;
    box-shadow: 0 0 6px #a78bfa;
    animation: livePulse 2s ease-in-out infinite;
}

.nx-h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(3rem, 7.5vw, 5.8rem);
    font-weight: 800;
    line-height: 0.98;
    letter-spacing: -0.05em;
    color: #ffffff;
    margin-bottom: 1.6rem;
}
.nx-h1 em {
    font-style: normal;
    background: linear-gradient(100deg, #a78bfa 5%, #38bdf8 45%, #34d399 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.nx-sub {
    font-size: 1.05rem;
    color: #475569;
    line-height: 1.8;
    max-width: 520px;
    margin: 0 auto;
    font-weight: 400;
}

/* ── Stat Cards (rendered via st.columns) ── */
.stat-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.055);
    border-radius: 18px;
    padding: 1.4rem 0.8rem 1.2rem;
    text-align: center;
    position: relative; overflow: hidden;
    transition: border-color 0.35s, transform 0.35s;
}
.stat-card:hover {
    border-color: rgba(139,92,246,0.35);
    transform: translateY(-3px);
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    width: 60%; height: 1px;
    background: linear-gradient(90deg, transparent, var(--sc, rgba(139,92,246,0.5)), transparent);
}
.stat-icon { font-size: 1.3rem; margin-bottom: 0.55rem; display: block; }
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    letter-spacing: -0.04em; line-height: 1;
    color: #fff; margin-bottom: 0.3rem;
}
.stat-lbl {
    font-size: 0.63rem; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #334155;
}

/* divider */
.nx-hr {
    height: 1px; margin: 0.5rem 0 3rem;
    background: linear-gradient(90deg, transparent 0%,
        rgba(124,58,237,0.45) 30%, rgba(56,189,248,0.35) 70%, transparent 100%);
}

/* ══════════════════════════════════════════════════════════════════
   SEARCH CARD
══════════════════════════════════════════════════════════════════ */
.nx-card {
    background: rgba(255,255,255,0.016);
    border: 1px solid rgba(255,255,255,0.065);
    border-radius: 24px;
    padding: 2.4rem 2.6rem 2rem;
    position: relative; overflow: hidden;
    backdrop-filter: blur(20px);
    margin-bottom: 1.8rem;
}
/* Animated gradient top border */
.nx-card::before {
    content: '';
    position: absolute; top: 0; left: -100%; right: -100%; height: 1px;
    background: linear-gradient(90deg, transparent 15%,
        rgba(124,58,237,0.7) 40%, rgba(56,189,248,0.7) 60%, transparent 85%);
    animation: borderSlide 4s linear infinite;
}
@keyframes borderSlide {
    0%   { transform: translateX(-60%); }
    100% { transform: translateX(60%); }
}

.card-lbl {
    font-size: 0.63rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #334155; margin-bottom: 0.9rem;
}

/* ── Text input ── */
.stTextInput > label { display: none !important; }
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    padding: 1rem 1.4rem !important;
    caret-color: #818cf8 !important;
    transition: all 0.3s ease !important;
}
.stTextInput > div > div > input::placeholder { color: #1e293b !important; }
.stTextInput > div > div > input:focus {
    border-color: rgba(129,140,248,0.5) !important;
    background: rgba(129,140,248,0.04) !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.1), 0 0 30px rgba(129,140,248,0.1) !important;
    outline: none !important;
}

/* ── Launch button ── */
.run-wrap > div > button,
.run-wrap button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important; font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    color: #fff !important;
    background: linear-gradient(135deg, #6d28d9 0%, #4338ca 55%, #0369a1 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.9rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 6px 32px rgba(109,40,217,0.45),
                0 0 0 1px rgba(255,255,255,0.06) inset !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
}
.run-wrap > div > button:hover,
.run-wrap button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 40px rgba(109,40,217,0.6),
                0 0 0 1px rgba(255,255,255,0.1) inset !important;
    filter: brightness(1.12) !important;
}
.run-wrap > div > button:active { transform: translateY(0) !important; }
.run-wrap button:disabled {
    background: rgba(255,255,255,0.04) !important;
    color: #1e293b !important;
    box-shadow: none !important; filter: none !important;
}

/* ── Chip buttons ── */
.chip > div > button, .chip button {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.065) !important;
    border-radius: 100px !important;
    color: #64748b !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important; font-weight: 500 !important;
    padding: 0.26rem 0.7rem !important;
    width: auto !important; min-width: unset !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
}
.chip > div > button:hover, .chip button:hover {
    background: rgba(124,58,237,0.12) !important;
    border-color: rgba(124,58,237,0.35) !important;
    color: #c4b5fd !important;
    transform: none !important; box-shadow: none !important;
}

/* ── Secondary / misc buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.83rem !important; font-weight: 500 !important;
    color: #64748b !important;
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.2rem !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    color: #e2e8f0 !important;
    background: rgba(255,255,255,0.055) !important;
    border-color: rgba(255,255,255,0.1) !important;
    transform: none !important; filter: none !important;
    box-shadow: none !important;
}

/* ══════════════════════════════════════════════════════════════════
   PIPELINE STEPS
══════════════════════════════════════════════════════════════════ */
.pipe-wrap { margin: 0.5rem 0 2.5rem; }
.pipe-lbl {
    font-size: 0.63rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #1e293b; margin-bottom: 1.1rem;
}
.pipe-row { display: flex; align-items: stretch; gap: 0; }

/* Each step card */
.ps {
    flex: 1; position: relative; overflow: hidden;
    background: rgba(255,255,255,0.014);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 1.5rem 1.1rem 1.3rem;
    transition: all 0.5s cubic-bezier(0.4,0,0.2,1);
}
/* Top accent bar */
.ps::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: rgba(255,255,255,0.04); transition: all 0.5s;
}
/* Connector chevron */
.ps-gap {
    width: 28px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
}
.ps-arr { font-size: 1rem; color: #1e293b; transition: color 0.4s; }

/* State classes */
.ps-idle {}
.ps-active {
    border-color: rgba(56,189,248,0.45);
    background: rgba(56,189,248,0.05);
    box-shadow: 0 0 40px rgba(56,189,248,0.1);
}
.ps-active::before {
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #38bdf8 100%);
    background-size: 200%;
    animation: sweep 1.8s linear infinite;
}
@keyframes sweep { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

.ps-done { border-color: rgba(52,211,153,0.35); background: rgba(52,211,153,0.03); }
.ps-done::before { background: linear-gradient(90deg, #10b981, #34d399); }
.ps-arr-done { color: #34d399; }

.ps-error { border-color: rgba(248,113,113,0.4); background: rgba(248,113,113,0.04); }
.ps-error::before { background: #f87171; }

/* Step number circle */
.ps-num {
    width: 22px; height: 22px; border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.1);
    font-size: 0.6rem; font-weight: 700; color: #334155;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 0.75rem;
}
.ps-icon { font-size: 1.35rem; display: block; margin-bottom: 0.5rem; }
.ps-name { font-size: 0.82rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.3rem; letter-spacing: -0.01em; }
.ps-desc { font-size: 0.68rem; color: #334155; line-height: 1.5; }
.ps-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    margin-top: 0.9rem;
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.2rem 0.55rem; border-radius: 5px;
}
.pb-idle  { color: #1e293b; background: rgba(30,41,59,0.4); }
.pb-active{ color: #38bdf8; background: rgba(56,189,248,0.12); }
.pb-active::before {
    content: ''; width: 4px; height: 4px; border-radius: 50%;
    background: #38bdf8; animation: livePulse 1s ease-in-out infinite;
}
.pb-done  { color: #34d399; background: rgba(52,211,153,0.12); }
.pb-error { color: #f87171; background: rgba(248,113,113,0.12); }

/* ══════════════════════════════════════════════════════════════════
   RESULTS
══════════════════════════════════════════════════════════════════ */
.res-head {
    display: flex; align-items: center; justify-content: space-between;
    margin: 3rem 0 1.4rem;
}
.res-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem; font-weight: 800;
    color: #fff; letter-spacing: -0.04em;
}
.res-tag {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: #34d399; padding: 0.28rem 0.85rem; border-radius: 100px;
    background: rgba(52,211,153,0.09);
    border: 1px solid rgba(52,211,153,0.22);
}

/* ── Tabs ── */
.stTabs [role="tablist"] {
    background: rgba(255,255,255,0.018) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 14px !important;
    padding: 0.28rem !important; gap: 0.1rem !important;
}
.stTabs [role="tab"] {
    background: transparent !important; border: none !important;
    border-radius: 10px !important; color: #334155 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.25s !important;
}
.stTabs [role="tab"]:hover { color: #64748b !important; background: rgba(255,255,255,0.025) !important; }
.stTabs [role="tab"][aria-selected="true"] {
    background: rgba(124,58,237,0.14) !important;
    color: #c4b5fd !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

/* ── Report reading area ── */
.report-wrap {
    background: rgba(255,255,255,0.012);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 2.8rem 3rem;
    color: #94a3b8;
    font-size: 0.93rem; line-height: 1.9;
}
.report-wrap h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.75rem; font-weight: 800; color: #f1f5f9;
    letter-spacing: -0.04em;
    margin: 0 0 1.2rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.report-wrap h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem; font-weight: 700; color: #e2e8f0;
    letter-spacing: -0.03em; margin: 2rem 0 0.7rem;
}
.report-wrap h3 { font-size: 0.98rem; font-weight: 600; color: #a78bfa; margin: 1.5rem 0 0.5rem; }
.report-wrap p  { margin-bottom: 1rem; }
.report-wrap ul, .report-wrap ol { padding-left: 1.4rem; margin-bottom: 1.1rem; }
.report-wrap li { margin-bottom: 0.45rem; }
.report-wrap strong { color: #f1f5f9; font-weight: 600; }
.report-wrap a { color: #38bdf8; text-decoration: none; }
.report-wrap a:hover { text-decoration: underline; }

/* ── Score display ── */
.score-wrap {
    background: rgba(255,255,255,0.016);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px; padding: 2rem 1.8rem;
    margin-bottom: 1.2rem; position: relative; overflow: hidden;
}
.score-bg-num {
    position: absolute; right: -10px; top: -20px;
    font-family: 'Syne', sans-serif; font-size: 9rem; font-weight: 800;
    color: rgba(255,255,255,0.025); line-height: 1; pointer-events: none;
    letter-spacing: -0.05em;
}
.score-val {
    font-family: 'Syne', sans-serif;
    font-size: 4.5rem; font-weight: 800;
    letter-spacing: -0.06em; line-height: 1;
}
.score-denom { font-size: 1.2rem; color: #1e293b; margin-left: 0.2rem; font-weight: 500; }
.score-lbl {
    font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase; color: #334155;
    margin-top: 0.4rem; margin-bottom: 1.4rem;
}
.gauge-track {
    height: 4px; background: rgba(255,255,255,0.04);
    border-radius: 10px; overflow: hidden;
}
.gauge-fill { height: 100%; border-radius: 10px; }

/* ── Critique boxes ── */
.crit-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.9rem; margin: 1.2rem 0; }
.crit-box {
    background: rgba(255,255,255,0.012);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px; padding: 1.4rem 1.3rem;
}
.crit-hd {
    font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.crit-body { font-size: 0.82rem; color: #475569; line-height: 1.75; }

/* ── Verdict card ── */
.verdict-card {
    background: rgba(255,255,255,0.012);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px; padding: 1.4rem 1.8rem; margin-top: 0.9rem;
}
.verdict-lbl {
    font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #334155; margin-bottom: 0.7rem;
}
.verdict-body { font-size: 0.85rem; color: #64748b; line-height: 1.8; }

/* ── Source cards ── */
.src-card {
    display: grid; grid-template-columns: 44px 1fr;
    gap: 1.1rem; align-items: start;
    background: rgba(255,255,255,0.014);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px; padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.25s, transform 0.25s;
}
.src-card:hover {
    border-color: rgba(124,58,237,0.28);
    transform: translateX(4px);
}
.src-idx {
    width: 38px; height: 38px; border-radius: 11px;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.18);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem; font-weight: 700; color: #a78bfa;
}
.src-title { font-size: 0.87rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.3rem; }
.src-url {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem; color: #38bdf8; text-decoration: none;
    display: block; margin-bottom: 0.5rem;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 580px;
}
.src-url:hover { text-decoration: underline; }
.src-snip { font-size: 0.79rem; color: #334155; line-height: 1.65; }

/* ── Download button ── */
.stDownloadButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    color: #818cf8 !important;
    background: rgba(129,140,248,0.07) !important;
    border: 1px solid rgba(129,140,248,0.2) !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.2rem !important;
    width: auto !important;
    box-shadow: none !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: rgba(129,140,248,0.14) !important;
    border-color: rgba(129,140,248,0.4) !important;
    transform: none !important; box-shadow: none !important;
}

/* ── Misc ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.3); border-radius: 10px; }
.stSpinner > div { border-top-color: #7c3aed !important; }
.stAlert { border-radius: 14px !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="bg-wrap">
    <div class="bg-blob bb1"></div>
    <div class="bg-blob bb2"></div>
    <div class="bg-blob bb3"></div>
</div>
<div class="bg-grid"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  NAV
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<nav class="nx-nav">
    <div class="nx-logo">
        <div class="nx-logo-mark">⚡</div>
        Nexus AI
    </div>
    <div class="nx-nav-center">
        <div class="nx-nav-tab on">Research</div>
        <div class="nx-nav-tab">History</div>
        <div class="nx-nav-tab">Settings</div>
    </div>
    <div class="nx-nav-live">
        <div class="live-dot"></div>
        All agents ready
    </div>
</nav>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for _k, _v in [("topic_input",""),("statuses",["idle"]*4),("pipeline_result",None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE OPEN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="nx-page">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="nx-hero">
    <div class="nx-eyebrow">
        <div class="eyebrow-dot"></div>
        Autonomous Multi-Agent Intelligence
    </div>
    <h1 class="nx-h1">
        Research anything.<br>
        <em>Instantly.</em>
    </h1>
    <p class="nx-sub">
        Four specialized AI agents work together in real-time — searching the web,
        reading sources, writing reports, and auditing quality. Zero effort from you.
    </p>
</div>
""", unsafe_allow_html=True)

# ── STAT CARDS via st.columns ──────────────────────────────────────────────
STATS = [
    ("⚡", "4",    "AI Agents",        "#7c3aed"),
    ("🌐", "5+",   "Sources Crawled",  "#0891b2"),
    ("⏱",  "~60s", "Avg Runtime",      "#059669"),
    ("🤖", "100%", "Autonomous",       "#7c3aed"),
]
st.markdown('<div style="height:2.5rem"></div>', unsafe_allow_html=True)
sc1, sc2, sc3, sc4 = st.columns(4, gap="small")
for col, (icon, num, lbl, color) in zip([sc1,sc2,sc3,sc4], STATS):
    with col:
        st.markdown(f"""
        <div class="stat-card" style="--sc:{color}80;">
            <span class="stat-icon">{icon}</span>
            <div class="stat-num" style="background:linear-gradient(135deg,{color},{color}aa);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
                {num}
            </div>
            <div class="stat-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="nx-hr"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH CARD
# ══════════════════════════════════════════════════════════════════════════════
PRESETS = [
    ("⚡","Fusion Energy 2025"), ("🚀","SpaceX Starship"), ("🧬","CRISPR Breakthroughs"),
    ("🤖","AI Reasoning Models"), ("🌊","Ocean Plastic Crisis"), ("💊","Weight Loss Drug Science"),
    ("🔋","Solid-State Batteries"), ("🌍","Climate Tech 2025"),
]

st.markdown('<div class="nx-card">', unsafe_allow_html=True)
st.markdown('<div class="card-lbl">Quick start — pick a topic</div>', unsafe_allow_html=True)

chip_cols = st.columns(len(PRESETS), gap="small")
for i, (em, lbl) in enumerate(PRESETS):
    with chip_cols[i]:
        st.markdown('<div class="chip">', unsafe_allow_html=True)
        if st.button(f"{em} {lbl}", key=f"chip_{i}"):
            st.session_state.topic_input = lbl
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:1.6rem"></div>', unsafe_allow_html=True)
st.markdown('<div class="card-lbl">Or describe your research question</div>', unsafe_allow_html=True)

topic = st.text_input(
    "topic", value=st.session_state.topic_input,
    placeholder="e.g.  Latest breakthroughs in quantum computing error correction",
    key="topic_field",
)
if topic != st.session_state.topic_input:
    st.session_state.topic_input = topic

st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
st.markdown('<div class="run-wrap">', unsafe_allow_html=True)
run_btn = st.button("⚡  Launch Research Pipeline", disabled=not bool(topic.strip()), key="run_btn")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)  # close nx-card

# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════
STEPS = [
    ("🔍", "Search Agent",  "Finds titles, URLs & summaries across 5+ sources via Tavily"),
    ("📖", "Reader Agent",  "Scrapes the top source and extracts full article content"),
    ("✍️", "Writer Chain",  "Drafts a structured, cited intelligence report"),
    ("🎯", "Critic Chain",  "Audits depth, accuracy & style — scores out of 10"),
]
S_MAP = {
    "idle":   ("ps-idle",   "pb-idle",   "Standby"),
    "active": ("ps-active", "pb-active", "Running"),
    "done":   ("ps-done",   "pb-done",   "Done ✓"),
    "error":  ("ps-error",  "pb-error",  "Failed"),
}

def render_pipe(statuses):
    h = '<div class="pipe-wrap"><p class="pipe-lbl">Agent Pipeline</p><div class="pipe-row">'
    for i, (icon, name, desc) in enumerate(STEPS):
        sc, bc, bt = S_MAP.get(statuses[i], S_MAP["idle"])
        h += f"""
        <div class="ps {sc}">
            <div class="ps-num">{i+1}</div>
            <span class="ps-icon">{icon}</span>
            <div class="ps-name">{name}</div>
            <div class="ps-desc">{desc}</div>
            <div class="ps-badge {bc}">{bt}</div>
        </div>"""
        if i < 3:
            ac = "ps-arr-done" if statuses[i] == "done" else \
                 "ps-arr-active" if statuses[i] == "active" else "ps-arr"
            h += f'<div class="ps-gap"><span class="{ac}">›</span></div>'
    h += '</div></div>'
    return h

pipe_ph = st.empty()
pipe_ph.markdown(render_pipe(st.session_state.statuses), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if run_btn and topic.strip():
    st.session_state.pipeline_result = None
    st.session_state.statuses = ["idle"] * 4
    state = {}; err = None

    def _set(idx, s):
        st.session_state.statuses[idx] = s
        pipe_ph.markdown(render_pipe(st.session_state.statuses), unsafe_allow_html=True)

    try:
        _set(0, "active")
        with st.spinner("Search Agent scanning the web…"):
            sa = build_search_agent()
            sr = sa.invoke({"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]})
            state["search_results"] = sr["messages"][-1].content
        _set(0, "done")

        _set(1, "active")
        with st.spinner("Reader Agent scraping sources…"):
            ra = build_reader_agent()
            rr = ra.invoke({"messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:800]}")]})
            state["scraped_content"] = rr["messages"][-1].content
        _set(1, "done")

        _set(2, "active")
        with st.spinner("Writer Chain composing report…"):
            state["report"] = writer_chain.invoke({
                "topic": topic,
                "research": f"SEARCH RESULTS:\n{state['search_results']}\n\nDETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
            })
        _set(2, "done")

        _set(3, "active")
        with st.spinner("Critic Chain evaluating quality…"):
            state["feedback"] = critic_chain.invoke({"report": state["report"]})
        _set(3, "done")

        st.session_state.pipeline_result = state

    except Exception as e:
        for i, s in enumerate(st.session_state.statuses):
            if s == "active": _set(i, "error")
        err = str(e)

    if err:
        st.error(f"Pipeline error: {err}")

# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.pipeline_result:
    res = st.session_state.pipeline_result

    st.markdown("""
    <div class="res-head">
        <div class="res-title">Research Report</div>
        <div class="res-tag">✦ Pipeline Complete</div>
    </div>
    """, unsafe_allow_html=True)

    tab_r, tab_c, tab_s = st.tabs(["📄  Report", "📊  Quality Audit", "🔗  Sources"])

    # ─── TAB 1: REPORT ────────────────────────────────────────────────────────
    with tab_r:
        report_md = res.get("report","")
        st.markdown('<div class="report-wrap">', unsafe_allow_html=True)
        st.markdown(report_md)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        st.download_button(
            "⬇  Download Report (.md)", data=report_md,
            file_name=f"nexus_{topic.lower().replace(' ','_')[:35]}.md",
            mime="text/markdown",
        )

    # ─── TAB 2: QUALITY AUDIT ─────────────────────────────────────────────────
    with tab_c:
        feedback = res.get("feedback","")
        sm = re.search(r"Score:\s*(\d+)", feedback, re.IGNORECASE)
        score = int(sm.group(1)) if sm else None

        col_sc, col_sp = st.columns([1, 2])
        with col_sc:
            if score is not None:
                if score >= 8:
                    sc, grd = "#34d399", "linear-gradient(90deg,#059669,#34d399,#6ee7b7)"
                elif score >= 6:
                    sc, grd = "#fbbf24", "linear-gradient(90deg,#d97706,#fbbf24,#fde68a)"
                else:
                    sc, grd = "#f87171", "linear-gradient(90deg,#dc2626,#f87171)"
                st.markdown(f"""
                <div class="score-wrap">
                    <div class="score-bg-num">{score}</div>
                    <div style="display:flex;align-items:baseline;gap:0.2rem">
                        <div class="score-val" style="color:{sc}">{score}</div>
                        <div class="score-denom">/ 10</div>
                    </div>
                    <div class="score-lbl">Quality Score</div>
                    <div class="gauge-track">
                        <div class="gauge-fill" style="width:{score*10}%;background:{grd}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="score-wrap">
                    <div class="score-val" style="color:#1e293b">—</div>
                    <div class="score-lbl">Score unavailable</div>
                </div>""", unsafe_allow_html=True)

        strn = re.findall(r"(?:Strengths:?)[:\s]+(.*?)(?=Weaknesses:|Verdict:|$)", feedback, re.DOTALL|re.IGNORECASE)
        weak = re.findall(r"(?:Weaknesses:?)[:\s]+(.*?)(?=Verdict:|Score:|$)",     feedback, re.DOTALL|re.IGNORECASE)
        vrdt = re.findall(r"(?:Verdict:?)[:\s]+(.*)",                              feedback, re.DOTALL|re.IGNORECASE)

        s_txt = strn[0].strip() if strn else "Well structured, comprehensive and informative."
        w_txt = weak[0].strip() if weak else "Consider expanding citations and numerical evidence."

        st.markdown(f"""
        <div class="crit-grid">
            <div class="crit-box">
                <div class="crit-hd" style="color:#34d399">✦ Strengths</div>
                <div class="crit-body">{s_txt.replace(chr(10),"<br>")}</div>
            </div>
            <div class="crit-box">
                <div class="crit-hd" style="color:#f87171">✦ Improvements</div>
                <div class="crit-body">{w_txt.replace(chr(10),"<br>")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        v_txt = vrdt[0].strip() if vrdt else (feedback if not strn and not weak else "")
        if v_txt:
            st.markdown(f"""
            <div class="verdict-card">
                <div class="verdict-lbl">Final Verdict</div>
                <div class="verdict-body">{v_txt.replace(chr(10),"<br>")}</div>
            </div>""", unsafe_allow_html=True)

    # ─── TAB 3: SOURCES ───────────────────────────────────────────────────────
    with tab_s:
        raw = res.get("search_results","")
        items = [x for x in raw.split("----") if x.strip()]

        st.markdown(
            f'<p style="font-size:0.62rem;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:#1e293b;margin-bottom:1.2rem;">'
            f'{len(items)} sources indexed</p>', unsafe_allow_html=True)

        for idx, item in enumerate(items, 1):
            tm = re.search(r"Title:\s*(.*)",   item, re.IGNORECASE)
            um = re.search(r"URL:\s*(.*)",     item, re.IGNORECASE)
            sm_s = re.search(r"Snippet:\s*(.*)", item, re.DOTALL|re.IGNORECASE)
            title   = tm.group(1).strip()         if tm   else "Referenced Article"
            url     = um.group(1).strip()         if um   else "#"
            snippet = sm_s.group(1).strip()[:230] if sm_s else item.strip()[:230]
            st.markdown(f"""
            <div class="src-card">
                <div class="src-idx">{idx:02d}</div>
                <div>
                    <div class="src-title">{title}</div>
                    <a class="src-url" href="{url}" target="_blank">{url}</a>
                    <div class="src-snip">{snippet}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:2rem"></div>', unsafe_allow_html=True)
        if st.button("↩  Start New Research", key="new_run"):
            st.session_state.pipeline_result = None
            st.session_state.statuses = ["idle"]*4
            st.session_state.topic_input = ""
            st.rerun()

# ─── Close page wrapper ───────────────────────────────────────────────────────
st.markdown('</div>', unsafe_allow_html=True)
