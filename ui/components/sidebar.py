"""
Enhanced Sidebar Component for AURIX 2026.
Features:
- Role-Based Access Control (RBAC) navigation
- Grouped navigation (Core Audit, Intelligence, Labs, Admin)
- Floating Copilot FAB integration
- Theme toggle and stats
"""

import streamlit as st
from typing import List, Dict, Optional
from ui.styles.css_builder import get_current_theme
from ui.components.badges import render_badge
from modules.rbac import (
    UserRole,
    get_user_role,
    set_user_role,
    get_permitted_pages,
    can_access_page,
    get_role_config,
    ROLE_CONFIGS,
)


def render_logo():
    """Render AURIX logo with native Streamlit."""
    st.markdown("### 🛡️ AURIX")
    st.caption("v4.2 Excellence 2026")


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


def render_role_selector():
    """Render role selector dropdown."""
    t = get_current_theme()

    # Initialize role if not set
    if 'user_role' not in st.session_state:
        st.session_state.user_role = UserRole.FIELD_AUDITOR

    current_role = get_user_role()
    role_config = get_role_config(current_role)

    st.caption("👤 ROLE")

    role_options = list(ROLE_CONFIGS.keys())
    role_labels = {r: ROLE_CONFIGS[r].display_name for r in role_options}

    selected_role = st.selectbox(
        "Select Role",
        role_options,
        index=role_options.index(current_role),
        format_func=lambda x: role_labels[x],
        key="role_selector",
        label_visibility="collapsed"
    )

    if selected_role != current_role:
        set_user_role(selected_role)
        st.session_state.current_page = "📊 Dashboard"  # Reset to dashboard on role change
        st.rerun()

    # Show role objective
    st.markdown(f"""
    <div style="font-size:0.7rem; color:{t['text_muted']}; padding:0.25rem 0;">
        {role_config.strategic_objective}
    </div>
    """, unsafe_allow_html=True)


# ============================================
# COMPACT NAVIGATION - Premium 2026
# ============================================

# Simplified navigation structure
NAVIGATION_SECTIONS = {
    "MAIN": [
        "📊 Dashboard",
        "🏛️ Executive Dashboard",
        "🎛️ Command Center",
        "📋 Findings Tracker",
    ],
    "AUDIT": [
        "📁 Documents",
        "📝 Workpapers",
        "⚖️ Risk Assessment",
        "🎭 PTCF Builder",
        "📅 Audit Planning",
    ],
    "RISK": [
        "📈 KRI Dashboard",
        "🎰 Stress Tester",
        "📊 IJK Benchmarking",
        "🔍 Fraud Detection",
    ],
    "ANALYTICS": [
        "🔄 Continuous Audit",
        "📊 Analytics",
        "🔄 Process Mining",
        "📜 Regulatory RAG",
    ],
    "MORE": [
        "🌐 Risk Universe",
        "🔬 Root Cause Analyzer",
        "🧪 AI Lab",
        "📑 Report Builder",
        "👥 Team Hub",
        "🔧 Admin Panel",
        "⚙️ Settings",
    ],
}


def render_compact_navigation(routes: List[str]) -> str:
    """Render compact navigation with tabs and selectbox, filtered by role."""
    t = get_current_theme()
    gold = t.get('gold', t['accent'])

    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📊 Dashboard"

    # Get permitted pages for current role
    current_role = get_user_role()
    permitted_pages = set(get_permitted_pages(current_role))

    # Section selector
    section_labels = {
        "MAIN": "🎯 Main",
        "AUDIT": "📋 Audit",
        "RISK": "⚠️ Risk",
        "ANALYTICS": "📊 Analytics",
        "MORE": "⚙️ More",
    }

    # Filter sections that have at least one permitted page
    active_sections = {}
    for section, pages in NAVIGATION_SECTIONS.items():
        section_pages = [p for p in pages if p in permitted_pages or p in routes]
        if section_pages:
            active_sections[section] = section_pages

    if not active_sections:
        st.warning("No pages available for your role")
        return "📊 Dashboard"

    # Find current section
    current_section = "MAIN"
    for section, pages in active_sections.items():
        if st.session_state.current_page in pages:
            current_section = section
            break

    # If current page is not permitted, reset to dashboard
    if st.session_state.current_page not in permitted_pages and st.session_state.current_page not in routes:
        st.session_state.current_page = "📊 Dashboard"
        current_section = "MAIN"

    # Section tabs - only show sections with permitted pages
    sections = [s for s in section_labels.keys() if s in active_sections]

    if len(sections) > 1:
        cols = st.columns(len(sections))
        for i, (col, section) in enumerate(zip(cols, sections)):
            with col:
                is_active = section == current_section
                if st.button(
                    section_labels[section].split()[0],  # Just emoji
                    key=f"sec_{section}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    # Switch to first page of section
                    available = active_sections.get(section, [])
                    if available:
                        st.session_state.current_page = available[0]
                        st.rerun()

    # Page list for current section (filtered by role)
    available_pages = active_sections.get(current_section, [])

    if available_pages:
        # Ensure current page is in available list
        current_index = 0
        if st.session_state.current_page in available_pages:
            current_index = available_pages.index(st.session_state.current_page)

        selected = st.radio(
            "Navigate",
            available_pages,
            index=current_index,
            key="nav_radio",
            label_visibility="collapsed"
        )

        if selected != st.session_state.current_page:
            st.session_state.current_page = selected
            st.rerun()

    return st.session_state.current_page


def render_grouped_navigation(routes: List[str], categories: Dict[str, List[str]]) -> str:
    """Wrapper for backward compatibility."""
    return render_compact_navigation(routes)


def render_session_stats():
    """Render session statistics using native Streamlit."""
    doc_count = len(st.session_state.get('documents', []))
    finding_count = len(st.session_state.get('findings', []))

    st.caption("📊 QUICK STATS")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Docs", doc_count)
    with col2:
        st.metric("Findings", finding_count)
    with col3:
        st.metric("Alerts", 3)


def render_llm_config():
    """Render LLM configuration section."""
    from app.constants import LLM_PROVIDER_INFO

    st.caption("🤖 AI PROVIDER")

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
        st.link_button("Get API Key →", info['url'], use_container_width=True)


def render_sidebar(routes: List[str], categories: Dict[str, List[str]]) -> str:
    """Render compact sidebar with navigation and RBAC."""
    with st.sidebar:
        render_logo()
        render_theme_toggle()

        st.divider()

        # Role selector - RBAC
        render_role_selector()

        st.divider()

        selected_page = render_grouped_navigation(routes, categories)

        st.divider()

        render_session_stats()

        st.divider()

        render_llm_config()

        st.caption("💡 Press 🤖 for AI Copilot")

    return selected_page
