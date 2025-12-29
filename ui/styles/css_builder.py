"""
CSS Builder for AURIX.
Generates dynamic CSS based on current theme.
Royal Purple & Gold Premium Theme.
"""

import streamlit as st
from typing import Dict
from app.constants import COLORS


def get_current_theme() -> Dict[str, str]:
    """Get current theme colors based on session state."""
    theme_name = st.session_state.get('theme', 'dark')
    return COLORS.get(theme_name, COLORS['dark'])


def inject_css():
    """Inject complete CSS styles into the page."""
    t = get_current_theme()
    is_dark = st.session_state.get('theme', 'dark') == 'dark'

    # Get gold color with fallback
    gold = t.get('gold', t['accent'])
    gold_light = t.get('gold_light', t['accent'])

    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ===== PREMIUM ANIMATIONS ===== */
@keyframes gold-shimmer {{
    0% {{ background-position: -200% center; }}
    100% {{ background-position: 200% center; }}
}}

@keyframes pulse-gold {{
    0% {{ box-shadow: 0 0 0 0 rgba(245, 182, 66, 0.4); }}
    70% {{ box-shadow: 0 0 0 10px rgba(245, 182, 66, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(245, 182, 66, 0); }}
}}

@keyframes gradient-shift {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

.gold-shimmer {{
    background: linear-gradient(90deg, transparent 0%, {gold}40 50%, transparent 100%);
    background-size: 200% 100%;
    animation: gold-shimmer 3s ease-in-out infinite;
}}

.pulse-gold {{
    animation: pulse-gold 2s infinite;
}}

/* ===== GLOBAL ===== */
* {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.stApp {{
    background: {t['bg']};
}}

/* ===== HIDE STREAMLIT DEFAULTS ===== */
#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{padding: 2rem 3rem 3rem 3rem; max-width: 1400px;}}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {{
    background: {t['sidebar_bg']};
    border-right: 1px solid {t['border']};
}}

section[data-testid="stSidebar"] .stRadio label {{
    color: {t['text']} !important;
}}

section[data-testid="stSidebar"] .stRadio label:hover {{
    background: {t['card_hover']};
    border-radius: 8px;
}}

/* ===== TYPOGRAPHY ===== */
h1, h2, h3, h4, h5, h6 {{
    color: {t['text']} !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}}

h1 {{ font-size: 2rem !important; }}
h2 {{ font-size: 1.5rem !important; margin-bottom: 1.5rem !important; }}
h3 {{ font-size: 1.125rem !important; }}

p, span, li, td, th, label {{
    color: {t['text_secondary']} !important;
}}

/* Light mode text visibility fix */
div[data-testid="stMarkdownContainer"] div {{
    color: {t['text']} !important;
}}

div[data-testid="stMarkdownContainer"] .metric-label,
div[data-testid="stMarkdownContainer"] .pro-card-header {{
    color: {t['text_muted']} !important;
}}

a {{
    color: {t['primary']} !important;
    text-decoration: none !important;
}}

a:hover {{
    color: {t['primary_hover']} !important;
}}

/* ===== CARDS - Premium Glass-morphism ===== */
.pro-card {{
    background: {'rgba(35, 28, 53, 0.85)' if is_dark else 'rgba(255, 255, 255, 0.95)'};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid {'rgba(139, 92, 246, 0.15)' if is_dark else 'rgba(124, 58, 237, 0.1)'};
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 24px {'rgba(13, 10, 26, 0.4)' if is_dark else 'rgba(0,0,0,0.06)'};
}}

.pro-card:hover {{
    border-color: {gold};
    box-shadow: 0 0 20px {'rgba(245, 182, 66, 0.15)' if is_dark else 'rgba(212, 160, 54, 0.1)'},
                0 8px 32px {'rgba(139, 92, 246, 0.2)' if is_dark else 'rgba(124, 58, 237, 0.1)'};
    transform: translateY(-2px);
}}

/* Premium Gold Border Accent */
.pro-card-gold {{
    border-top: 2px solid {gold};
}}

.pro-card-gold:hover {{
    border-top-color: {gold_light};
}}

.pro-card-header {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {t['text_muted']} !important;
    margin-bottom: 0.5rem;
}}

.pro-card-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {t['text']} !important;
    line-height: 1.2;
}}

/* ===== METRICS ===== */
.metric-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}}

@media (max-width: 1024px) {{
    .metric-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}

.metric-card {{
    background: {'rgba(35, 28, 53, 0.8)' if is_dark else 'rgba(255, 255, 255, 0.95)'};
    backdrop-filter: blur(8px);
    border: 1px solid {'rgba(139, 92, 246, 0.12)' if is_dark else 'rgba(124, 58, 237, 0.08)'};
    border-radius: 16px;
    padding: 1.25rem;
    text-align: left;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}}

.metric-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, {t['primary']}, {gold});
    opacity: 0;
    transition: opacity 0.3s ease;
}}

.metric-card:hover::before {{
    opacity: 1;
}}

.metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px {'rgba(139, 92, 246, 0.15)' if is_dark else 'rgba(124, 58, 237, 0.08)'};
}}

.metric-label {{
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {t['text_muted']} !important;
    margin-bottom: 0.5rem;
}}

.metric-value {{
    font-size: 1.75rem;
    font-weight: 700;
    color: {t['text']} !important;
}}

.metric-change {{
    font-size: 0.75rem;
    margin-top: 0.25rem;
}}

.metric-change.positive {{ color: {t['success']} !important; }}
.metric-change.negative {{ color: {t['danger']} !important; }}

/* ===== BADGES ===== */
.badge {{
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.025em;
}}

.badge-high, .badge-critical, .badge-danger {{
    background: {'rgba(239,68,68,0.15)' if is_dark else 'rgba(220,38,38,0.1)'};
    color: {t['danger']} !important;
    border: 1px solid {'rgba(239,68,68,0.3)' if is_dark else 'rgba(220,38,38,0.2)'};
}}

.badge-medium, .badge-warning {{
    background: {'rgba(245,158,11,0.15)' if is_dark else 'rgba(217,119,6,0.1)'};
    color: {t['warning']} !important;
    border: 1px solid {'rgba(245,158,11,0.3)' if is_dark else 'rgba(217,119,6,0.2)'};
}}

.badge-low, .badge-success {{
    background: {'rgba(16,185,129,0.15)' if is_dark else 'rgba(5,150,105,0.1)'};
    color: {t['success']} !important;
    border: 1px solid {'rgba(16,185,129,0.3)' if is_dark else 'rgba(5,150,105,0.2)'};
}}

.badge-open {{
    background: {'rgba(245,158,11,0.15)' if is_dark else 'rgba(217,119,6,0.1)'};
    color: {t['warning']} !important;
}}

.badge-closed {{
    background: {'rgba(16,185,129,0.15)' if is_dark else 'rgba(5,150,105,0.1)'};
    color: {t['success']} !important;
}}

/* ===== BUTTONS - Premium Styling ===== */
.stButton > button {{
    background: linear-gradient(135deg, {t['primary']} 0%, {t['primary_hover']} 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px {'rgba(139, 92, 246, 0.3)' if is_dark else 'rgba(124, 58, 237, 0.2)'} !important;
}}

.stButton > button:hover {{
    background: linear-gradient(135deg, {t['primary_hover']} 0%, {t['primary']} 100%) !important;
    box-shadow: 0 6px 20px {'rgba(139, 92, 246, 0.4)' if is_dark else 'rgba(124, 58, 237, 0.3)'},
                0 0 30px {'rgba(245, 182, 66, 0.15)' if is_dark else 'rgba(212, 160, 54, 0.1)'} !important;
    transform: translateY(-1px) !important;
}}

/* Gold Accent Button */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {t['primary']} 0%, {gold} 100%) !important;
}}

.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, {gold} 0%, {t['primary']} 100%) !important;
}}

/* ===== INPUTS - Premium ===== */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {{
    background: {'rgba(26, 20, 40, 0.6)' if is_dark else 'rgba(255, 255, 255, 0.95)'} !important;
    border: 1px solid {'rgba(139, 92, 246, 0.2)' if is_dark else 'rgba(124, 58, 237, 0.15)'} !important;
    border-radius: 10px !important;
    color: {t['text']} !important;
    transition: all 0.3s ease !important;
}}

.stTextInput input:focus, .stTextArea textarea:focus {{
    border-color: {gold} !important;
    box-shadow: 0 0 0 3px {'rgba(245, 182, 66, 0.15)' if is_dark else 'rgba(212, 160, 54, 0.1)'},
                0 4px 12px {'rgba(139, 92, 246, 0.15)' if is_dark else 'rgba(124, 58, 237, 0.08)'} !important;
}}

/* ===== TABLES ===== */
.pro-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.875rem;
}}

.pro-table th {{
    background: {t['bg_secondary']};
    color: {t['text_muted']} !important;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid {t['border']};
}}

.pro-table td {{
    padding: 0.875rem 1rem;
    border-bottom: 1px solid {t['border']};
    color: {t['text_secondary']} !important;
}}

.pro-table tr:hover td {{
    background: {t['card_hover']};
}}

/* ===== ALERTS ===== */
.alert-box {{
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid;
    margin: 1rem 0;
}}

.alert-danger {{
    background: {'rgba(239,68,68,0.1)' if is_dark else 'rgba(220,38,38,0.05)'};
    border-color: {t['danger']};
    color: {t['danger']} !important;
}}

.alert-warning {{
    background: {'rgba(245,158,11,0.1)' if is_dark else 'rgba(217,119,6,0.05)'};
    border-color: {t['warning']};
    color: {t['warning']} !important;
}}

.alert-success {{
    background: {'rgba(16,185,129,0.1)' if is_dark else 'rgba(5,150,105,0.05)'};
    border-color: {t['success']};
    color: {t['success']} !important;
}}

/* ===== LIST ITEMS ===== */
.list-item {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.875rem 0;
    border-bottom: 1px solid {t['border']};
}}

.list-item:last-child {{
    border-bottom: none;
}}

.list-item-title {{
    font-weight: 500;
    color: {t['text']} !important;
}}

.list-item-subtitle {{
    font-size: 0.8rem;
    color: {t['text_muted']} !important;
}}

/* ===== PROGRESS BAR ===== */
.progress-bar {{
    width: 100%;
    height: 8px;
    background: {t['border']};
    border-radius: 4px;
    overflow: hidden;
    margin: 0.5rem 0;
}}

.progress-fill {{
    height: 100%;
    background: {t['primary']};
    transition: width 0.3s ease;
}}

/* ===== STAT CARD ===== */
.stat-card {{
    background: {t['card']};
    border: 1px solid {t['border']};
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}}

.stat-icon {{
    font-size: 2rem;
    margin-bottom: 0.5rem;
}}

.stat-value {{
    font-size: 1.75rem;
    font-weight: 700;
    color: {t['text']} !important;
}}

.stat-label {{
    font-size: 0.75rem;
    color: {t['text_muted']} !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}}

/* ===== SECTION TITLE ===== */
.section-title {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {t['text_muted']} !important;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {t['border']};
}}

/* ===== LOGO ===== */
.logo-container {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1.25rem;
    border-bottom: 1px solid {t['border']};
    margin-bottom: 1rem;
}}

.logo-text {{
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: {t['text']} !important;
}}

.logo-tagline {{
    font-size: 0.65rem;
    color: {t['text_muted']} !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}}

/* ===== CHAT ===== */
.chat-user {{
    background: {t['primary']};
    color: white !important;
    padding: 1rem;
    border-radius: 12px 12px 4px 12px;
    margin: 0.5rem 0;
    margin-left: 15%;
}}

.chat-ai {{
    background: {t['card']};
    border: 1px solid {t['border']};
    padding: 1rem;
    border-radius: 12px 12px 12px 4px;
    margin: 0.5rem 0;
    margin-right: 15%;
}}

/* ===== FOOTER ===== */
.pro-footer {{
    text-align: center;
    padding: 2rem 0;
    margin-top: 3rem;
    border-top: 1px solid {t['border']};
}}

.footer-brand {{
    font-size: 1rem;
    font-weight: 700;
    color: {t['text']} !important;
    margin-bottom: 0.25rem;
}}

.footer-tagline {{
    font-size: 0.75rem;
    color: {t['text_muted']} !important;
    margin-bottom: 1rem;
}}

.footer-disclaimer {{
    background: {'rgba(245,158,11,0.1)' if is_dark else 'rgba(217,119,6,0.05)'};
    border: 1px solid {'rgba(245,158,11,0.2)' if is_dark else 'rgba(217,119,6,0.15)'};
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.75rem;
    color: {t['text_muted']} !important;
    max-width: 800px;
    margin: 0 auto 1rem auto;
    text-align: left;
}}

/* ===== KRI GAUGE ===== */
.kri-gauge {{
    width: 100%;
    height: 120px;
    position: relative;
    margin: 1rem 0;
}}

.kri-gauge-bg {{
    width: 100%;
    height: 15px;
    background: linear-gradient(to right, {t['success']}, {t['warning']}, {t['danger']});
    border-radius: 10px;
    position: relative;
}}

.kri-gauge-pointer {{
    width: 4px;
    height: 30px;
    background: {t['text']};
    position: absolute;
    top: -7.5px;
    transform: translateX(-2px);
}}

.kri-gauge-value {{
    text-align: center;
    margin-top: 0.5rem;
    font-size: 1.5rem;
    font-weight: 700;
    color: {t['text']} !important;
}}

/* ===== SCROLLBAR - Premium ===== */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}

::-webkit-scrollbar-track {{
    background: {t['bg']};
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb {{
    background: linear-gradient(180deg, {t['primary']}60, {t['border']});
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: linear-gradient(180deg, {gold}80, {t['primary']}60);
}}

/* ===== PREMIUM GRADIENT HEADERS ===== */
.premium-header {{
    background: linear-gradient(135deg, {t['primary']} 0%, {t['primary_hover']} 50%, {gold} 100%);
    background-size: 200% 200%;
    animation: gradient-shift 8s ease infinite;
    color: white !important;
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px {'rgba(139, 92, 246, 0.3)' if is_dark else 'rgba(124, 58, 237, 0.2)'};
}}

.premium-header h1, .premium-header h2, .premium-header h3,
.premium-header p, .premium-header span {{
    color: white !important;
}}

/* ===== GOLD ACCENTS ===== */
.gold-text {{
    color: {gold} !important;
    font-weight: 600;
}}

.gold-border {{
    border: 2px solid {gold} !important;
}}

.gold-glow {{
    box-shadow: 0 0 20px {'rgba(245, 182, 66, 0.3)' if is_dark else 'rgba(212, 160, 54, 0.2)'};
}}

/* ===== PREMIUM BADGES ===== */
.badge-premium {{
    background: linear-gradient(135deg, {gold} 0%, {gold_light} 100%);
    color: #1a1028 !important;
    font-weight: 700;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    box-shadow: 0 2px 8px rgba(245, 182, 66, 0.3);
}}

/* ===== TABS - Premium ===== */
.stTabs [data-baseweb="tab-list"] {{
    background: {'rgba(26, 20, 40, 0.5)' if is_dark else 'rgba(245, 243, 250, 0.8)'};
    border-radius: 12px;
    padding: 0.25rem;
    gap: 0.25rem;
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 10px;
    color: {t['text_secondary']} !important;
    font-weight: 500;
    transition: all 0.3s ease;
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {t['primary']} 0%, {t['primary_hover']} 100%) !important;
    color: white !important;
    box-shadow: 0 2px 8px {'rgba(139, 92, 246, 0.3)' if is_dark else 'rgba(124, 58, 237, 0.2)'};
}}

/* ===== EXPANDERS - Premium ===== */
.streamlit-expanderHeader {{
    background: {'rgba(35, 28, 53, 0.6)' if is_dark else 'rgba(255, 255, 255, 0.9)'} !important;
    border: 1px solid {'rgba(139, 92, 246, 0.15)' if is_dark else 'rgba(124, 58, 237, 0.1)'} !important;
    border-radius: 12px !important;
}}

/* ===== DIVIDERS - Premium ===== */
hr {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, {t['border']}, {gold}30, {t['border']}, transparent);
    margin: 2rem 0;
}}

/* ===== SELECT BOXES - Premium ===== */
.stSelectbox [data-baseweb="select"] {{
    background: {'rgba(26, 20, 40, 0.6)' if is_dark else 'rgba(255, 255, 255, 0.95)'} !important;
    border-radius: 10px !important;
}}

/* ===== SLIDERS - Premium ===== */
.stSlider [data-baseweb="slider"] {{
    background: transparent !important;
}}

.stSlider [role="slider"] {{
    background: {gold} !important;
    border: 2px solid {t['primary']} !important;
}}

/* ===== MULTISELECT - Premium ===== */
.stMultiSelect [data-baseweb="tag"] {{
    background: linear-gradient(135deg, {t['primary']}40, {gold}20) !important;
    border: 1px solid {t['primary']}60 !important;
    border-radius: 8px !important;
}}
</style>
"""
    
    st.markdown(css, unsafe_allow_html=True)


def get_color(color_name: str) -> str:
    """Get a specific color from current theme."""
    t = get_current_theme()
    return t.get(color_name, "#000000")
