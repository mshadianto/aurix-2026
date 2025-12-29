"""
Enhanced Sidebar Component for AURIX 2026.
Features:
- Grouped navigation (Core Audit, Intelligence, Labs, Admin)
- Floating Copilot FAB integration
- Theme toggle and stats
"""

import streamlit as st
from typing import List, Dict, Optional
from ui.styles.css_builder import get_current_theme
from ui.components.badges import render_badge


def get_logo_svg() -> str:
    """Generate AURIX logo SVG with Royal Purple & Gold theme."""
    t = get_current_theme()
    primary = t['primary']
    gold = t.get('gold', t['accent'])

    return f'''<svg width="36" height="36" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<defs>
    <linearGradient id="shieldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:{primary};stop-opacity:1" />
        <stop offset="100%" style="stop-color:{gold};stop-opacity:1" />
    </linearGradient>
    <filter id="glow">
        <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
        <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
        </feMerge>
    </filter>
</defs>
<path d="M50 8 L90 25 L90 55 C90 78 70 92 50 98 C30 92 10 78 10 55 L10 25 Z" fill="url(#shieldGrad)" filter="url(#glow)"/>
<circle cx="50" cy="48" r="8" fill="{gold}" opacity="0.95"/>
<circle cx="35" cy="33" r="4" fill="white" opacity="0.85"/>
<circle cx="65" cy="33" r="4" fill="white" opacity="0.85"/>
<circle cx="30" cy="55" r="4" fill="white" opacity="0.85"/>
<circle cx="70" cy="55" r="4" fill="white" opacity="0.85"/>
<circle cx="50" cy="72" r="4" fill="{gold}" opacity="0.9"/>
<line x1="50" y1="48" x2="35" y2="33" stroke="{gold}" stroke-width="2" opacity="0.6"/>
<line x1="50" y1="48" x2="65" y2="33" stroke="{gold}" stroke-width="2" opacity="0.6"/>
<line x1="50" y1="48" x2="30" y2="55" stroke="white" stroke-width="2" opacity="0.5"/>
<line x1="50" y1="48" x2="70" y2="55" stroke="white" stroke-width="2" opacity="0.5"/>
<line x1="50" y1="48" x2="50" y2="72" stroke="{gold}" stroke-width="2" opacity="0.7"/>
</svg>'''


def render_logo():
    """Render AURIX logo with premium branding."""
    t = get_current_theme()
    gold = t.get('gold', t['accent'])

    st.markdown(f'''
    <div class="logo-container" style="background:linear-gradient(135deg, {t['card']} 0%, {t['bg_secondary']} 100%);border-bottom:1px solid {gold}30;">
        {get_logo_svg()}
        <div>
            <div class="logo-text" style="background:linear-gradient(135deg, {t['text']} 0%, {gold} 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">AURIX</div>
            <div class="logo-tagline" style="color:{gold} !important;">v4.2 Excellence 2026</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def render_theme_toggle():
    """Render theme toggle buttons."""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("☀️ Light", use_container_width=True, key="light_btn"):
            st.session_state.theme = 'light'
            st.rerun()
    
    with col2:
        if st.button("🌙 Dark", use_container_width=True, key="dark_btn"):
            st.session_state.theme = 'dark'
            st.rerun()


# ============================================
# GROUPED NAVIGATION - 2026 Enhancement
# ============================================

# Define navigation groups with their pages
NAVIGATION_GROUPS = {
    "🎯 Core Audit": {
        "expanded": True,
        "badge": None,
        "pages": [
            "📊 Dashboard",
            "🎛️ Command Center",
            "📁 Documents",
            "🎭 PTCF Builder",
            "⚖️ Risk Assessment",
            "📋 Findings Tracker",
            "📝 Workpapers",
        ]
    },
    "🔬 Audit Tools": {
        "expanded": False,
        "badge": None,
        "pages": [
            "🌐 Risk Universe",
            "📌 Issue Tracker",
            "📅 Audit Planning",
            "📆 Audit Timeline",
            "🔬 Root Cause Analyzer",
            "🧮 Sampling Calculator",
        ]
    },
    "🧠 Intelligence": {
        "expanded": True,
        "badge": "3 Alerts",
        "pages": [
            "🔄 Continuous Audit",
            "📈 KRI Dashboard",
            "🔍 Fraud Detection",
            "🔄 Process Mining",
            "📜 Regulatory RAG",
            "📊 Analytics",
        ]
    },
    "🧪 Labs": {
        "expanded": False,
        "badge": "Beta",
        "pages": [
            "🧪 AI Lab",
            "📑 Report Builder",
        ]
    },
    "👥 Collaboration": {
        "expanded": False,
        "badge": None,
        "pages": [
            "👥 Team Hub",
            "🎮 Gamification",
        ]
    },
    "⚙️ Admin": {
        "expanded": False,
        "badge": None,
        "pages": [
            "📚 Regulations",
            "⚙️ Settings",
            "❓ Help",
            "ℹ️ About",
        ]
    }
}


def render_grouped_navigation(routes: List[str], categories: Dict[str, List[str]]) -> str:
    """
    Render grouped navigation with expandable sections.
    """
    t = get_current_theme()
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📊 Dashboard"
    
    selected_page = st.session_state.current_page
    
    for group_name, group_config in NAVIGATION_GROUPS.items():
        available_pages = [p for p in group_config["pages"] if p in routes or p in ["🔄 Process Mining", "📜 Regulatory RAG"]]
        
        if not available_pages:
            continue
        
        with st.expander(f"**{group_name}**", expanded=group_config["expanded"]):
            # Show badge if exists
            if group_config.get("badge"):
                badge_color = '#DC3545' if 'Alert' in str(group_config['badge']) else '#6C757D'
                st.markdown(f'''
                <span style="background:{badge_color};color:white;font-size:0.65rem;padding:0.15rem 0.5rem;border-radius:8px;">
                    {group_config['badge']}
                </span>
                ''', unsafe_allow_html=True)
            
            for page_name in available_pages:
                is_active = selected_page == page_name
                button_type = "primary" if is_active else "secondary"
                
                if st.button(
                    page_name,
                    key=f"nav_{page_name}",
                    use_container_width=True,
                    type=button_type
                ):
                    st.session_state.current_page = page_name
                    selected_page = page_name
                    st.rerun()
    
    return selected_page


def render_session_stats():
    """Render session statistics with premium styling."""
    t = get_current_theme()
    gold = t.get('gold', t['accent'])

    doc_count = len(st.session_state.get('documents', []))
    finding_count = len(st.session_state.get('findings', []))

    st.markdown(f'''
    <div class="pro-card pro-card-gold" style="padding:1rem;">
        <div style="font-size:0.7rem;color:{gold};text-transform:uppercase;letter-spacing:1px;margin-bottom:0.75rem;font-weight:600;">Quick Stats</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;padding:0.25rem 0;">
            <span style="color:{t['text_muted']} !important;font-size:0.8rem;">Documents</span>
            <span style="color:{t['text']} !important;font-weight:700;">{doc_count}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;padding:0.25rem 0;">
            <span style="color:{t['text_muted']} !important;font-size:0.8rem;">Findings</span>
            <span style="color:{t['danger']} !important;font-weight:700;">{finding_count}</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:0.25rem 0;">
            <span style="color:{t['text_muted']} !important;font-size:0.8rem;">KRI Alerts</span>
            <span style="color:{t['warning']} !important;font-weight:700;">3</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def render_llm_config():
    """Render LLM configuration section."""
    from app.constants import LLM_PROVIDER_INFO
    
    st.markdown('<div class="section-title">AI Provider</div>', unsafe_allow_html=True)
    
    providers = list(LLM_PROVIDER_INFO.keys())
    
    provider = st.selectbox(
        "Provider",
        providers,
        key="llm_provider",
        label_visibility="collapsed",
        format_func=lambda x: f"{LLM_PROVIDER_INFO[x]['name']} {'🆓' if LLM_PROVIDER_INFO[x]['free'] else '💎'}"
    )
    
    info = LLM_PROVIDER_INFO.get(provider, {})
    
    if provider not in ['mock', 'ollama']:
        st.text_input(
            "API Key",
            type="password",
            key="api_key_input",
            label_visibility="collapsed",
            placeholder="Enter API key..."
        )
    
    if info.get('url'):
        st.markdown(
            f"<small><a href='{info['url']}' target='_blank'>Get API Key →</a></small>",
            unsafe_allow_html=True
        )


def render_sidebar(routes: List[str], categories: Dict[str, List[str]]) -> str:
    """
    Render complete sidebar with grouped navigation.
    """
    t = get_current_theme()

    with st.sidebar:
        render_logo()
        render_theme_toggle()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        selected_page = render_grouped_navigation(routes, categories)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        render_session_stats()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        render_llm_config()
        
        gold = t.get('gold', t['accent'])
        st.markdown(f'''
        <div style="margin-top:1rem;padding:0.75rem;background:linear-gradient(135deg, {t['primary']}15, {gold}10);border:1px solid {t['primary']}20;border-radius:10px;text-align:center;">
            <span style="font-size:0.75rem;color:{t['text_muted']};">
                💡 Press <strong style="color:{gold};">🤖</strong> button for AI Copilot
            </span>
        </div>
        ''', unsafe_allow_html=True)

    return selected_page
