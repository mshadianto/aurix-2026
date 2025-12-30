"""
Admin Panel Page for AURIX 2026.
System administration, user management, and AI performance monitoring.
Only accessible by System Admin role.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List
import random

from ui.styles.css_builder import get_current_theme
from ui.components import render_page_header, render_footer
from modules.rbac import (
    UserRole,
    get_user_role,
    ROLE_CONFIGS,
    get_all_roles_info,
    can_access_page,
)


def render():
    """Render the Admin Panel page."""
    t = get_current_theme()

    # Check access
    current_role = get_user_role()
    if current_role != UserRole.SYSTEM_ADMIN:
        st.error("Access Denied: This page is only accessible by System Administrators.")
        return

    render_page_header(
        "Admin Panel",
        "System configuration, security monitoring, and AI performance"
    )

    # Admin tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 User Management",
        "🔐 Security",
        "🤖 AI Performance",
        "🗄️ Database",
        "⚙️ System Config"
    ])

    with tab1:
        _render_user_management(t)

    with tab2:
        _render_security_tab(t)

    with tab3:
        _render_ai_performance(t)

    with tab4:
        _render_database_tab(t)

    with tab5:
        _render_system_config(t)

    render_footer()


def _render_user_management(t: dict):
    """Render user management section."""
    st.markdown("### 👥 User Management")

    # User statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Users", "156", "+12 this month")
    with col2:
        st.metric("Active Today", "45", "29%")
    with col3:
        st.metric("Pending Approval", "3")
    with col4:
        st.metric("Locked Accounts", "2")

    st.markdown("---")

    # Role distribution
    st.markdown("#### Role Distribution")

    role_counts = {
        "Executive / Board": 8,
        "Audit Manager": 15,
        "Field Auditor": 98,
        "Auditee": 32,
        "System Admin": 3,
    }

    for role_name, count in role_counts.items():
        pct = count / sum(role_counts.values()) * 100
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.5rem 0;">
            <span style="color:{t['text']};">{role_name}</span>
            <div style="display:flex; align-items:center; gap:1rem;">
                <div style="width:200px; background:{t['border']}; height:8px; border-radius:4px;">
                    <div style="width:{pct}%; background:{t['primary']}; height:100%; border-radius:4px;"></div>
                </div>
                <span style="color:{t['text']}; font-weight:600; min-width:40px;">{count}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # User table (mock)
    st.markdown("---")
    st.markdown("#### Recent Users")

    users = [
        {"name": "John Doe", "email": "john@company.com", "role": "Field Auditor", "status": "Active", "last_login": "2024-12-30 09:15"},
        {"name": "Jane Smith", "email": "jane@company.com", "role": "Audit Manager", "status": "Active", "last_login": "2024-12-30 08:45"},
        {"name": "Bob Wilson", "email": "bob@company.com", "role": "Executive", "status": "Active", "last_login": "2024-12-29 16:30"},
        {"name": "Alice Brown", "email": "alice@company.com", "role": "Auditee", "status": "Pending", "last_login": "-"},
    ]

    for user in users:
        status_color = t['success'] if user['status'] == 'Active' else t['warning']
        st.markdown(f"""
        <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:8px; padding:0.75rem; margin:0.5rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-weight:600; color:{t['text']};">{user['name']}</div>
                    <div style="font-size:0.8rem; color:{t['text_muted']};">{user['email']}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.85rem; color:{t['text']};">{user['role']}</div>
                </div>
                <div style="text-align:center;">
                    <span style="background:{status_color}20; color:{status_color}; padding:0.25rem 0.5rem; border-radius:4px; font-size:0.8rem;">
                        {user['status']}
                    </span>
                </div>
                <div style="text-align:right; font-size:0.8rem; color:{t['text_muted']};">
                    {user['last_login']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_security_tab(t: dict):
    """Render security monitoring section."""
    st.markdown("### 🔐 Security Monitoring")

    # Security metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Failed Logins (24h)", "12", "-5 vs yesterday")
    with col2:
        st.metric("Active Sessions", "45")
    with col3:
        st.metric("API Calls (24h)", "15,842")
    with col4:
        st.metric("Security Score", "94/100")

    st.markdown("---")

    # Recent security events
    st.markdown("#### Recent Security Events")

    events = [
        {"time": "10:45", "type": "Login Failed", "user": "unknown@test.com", "ip": "192.168.1.100", "severity": "Warning"},
        {"time": "10:32", "type": "Password Reset", "user": "john@company.com", "ip": "10.0.0.15", "severity": "Info"},
        {"time": "09:15", "type": "Role Changed", "user": "jane@company.com", "ip": "10.0.0.22", "severity": "Info"},
        {"time": "08:45", "type": "Suspicious Activity", "user": "bob@company.com", "ip": "203.45.67.89", "severity": "High"},
    ]

    for event in events:
        severity_colors = {
            "High": t['danger'],
            "Warning": t['warning'],
            "Info": t['text_muted'],
        }
        color = severity_colors.get(event['severity'], t['text_muted'])

        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.5rem 0; border-bottom:1px solid {t['border']};">
            <span style="color:{t['text_muted']}; min-width:60px;">{event['time']}</span>
            <span style="color:{t['text']}; flex:1; margin-left:1rem;">{event['type']}</span>
            <span style="color:{t['text_muted']}; flex:1;">{event['user']}</span>
            <span style="color:{t['text_muted']}; flex:1;">{event['ip']}</span>
            <span style="background:{color}20; color:{color}; padding:0.2rem 0.5rem; border-radius:4px; font-size:0.8rem;">
                {event['severity']}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Access control
    st.markdown("---")
    st.markdown("#### Access Control Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.checkbox("Require 2FA for all users", value=True)
        st.checkbox("Session timeout after 30 minutes", value=True)
        st.checkbox("Lock account after 5 failed attempts", value=True)

    with col2:
        st.checkbox("IP whitelist enabled", value=False)
        st.checkbox("Audit log retention (90 days)", value=True)
        st.checkbox("Data encryption at rest", value=True)


def _render_ai_performance(t: dict):
    """Render AI/LLM performance monitoring."""
    st.markdown("### 🤖 AI Performance Monitoring")

    # AI metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("LLM Calls (24h)", "2,847", "+15%")
    with col2:
        st.metric("Avg Latency", "1.2s", "-0.3s")
    with col3:
        st.metric("Success Rate", "99.2%")
    with col4:
        st.metric("Token Usage", "1.2M", "of 5M limit")

    st.markdown("---")

    # Provider performance
    st.markdown("#### Provider Performance")

    providers = [
        {"name": "OpenAI GPT-4", "calls": 1542, "latency": "1.1s", "success": "99.5%", "cost": "$45.20"},
        {"name": "Anthropic Claude", "calls": 892, "latency": "1.3s", "success": "99.1%", "cost": "$28.50"},
        {"name": "Ollama (Local)", "calls": 413, "latency": "0.8s", "success": "98.8%", "cost": "$0.00"},
    ]

    for provider in providers:
        st.markdown(f"""
        <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:8px; padding:0.75rem; margin:0.5rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="font-weight:600; color:{t['text']}; min-width:150px;">{provider['name']}</div>
                <div style="text-align:center;">
                    <div style="font-size:1.1rem; font-weight:600; color:{t['text']};">{provider['calls']}</div>
                    <div style="font-size:0.7rem; color:{t['text_muted']};">Calls</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.1rem; font-weight:600; color:{t['text']};">{provider['latency']}</div>
                    <div style="font-size:0.7rem; color:{t['text_muted']};">Latency</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.1rem; font-weight:600; color:{t['success']};">{provider['success']}</div>
                    <div style="font-size:0.7rem; color:{t['text_muted']};">Success</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1.1rem; font-weight:600; color:{t['accent']};">{provider['cost']}</div>
                    <div style="font-size:0.7rem; color:{t['text_muted']};">Cost (24h)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # RAG performance
    st.markdown("---")
    st.markdown("#### RAG Quality Metrics")

    rag_metrics = [
        ("Faithfulness Score", "0.92", t['success']),
        ("Context Relevance", "0.88", t['success']),
        ("Answer Relevance", "0.85", t['warning']),
        ("Hallucination Rate", "3.2%", t['success']),
    ]

    cols = st.columns(4)
    for col, (metric, value, color) in zip(cols, rag_metrics):
        with col:
            st.markdown(f"""
            <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:8px; padding:1rem; text-align:center;">
                <div style="font-size:1.5rem; font-weight:700; color:{color};">{value}</div>
                <div style="font-size:0.8rem; color:{t['text_muted']};">{metric}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_database_tab(t: dict):
    """Render database management section."""
    st.markdown("### 🗄️ Database Management")

    # Database metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", "1.2M")
    with col2:
        st.metric("Storage Used", "4.5 GB", "of 20 GB")
    with col3:
        st.metric("Query Latency", "45ms")
    with col4:
        st.metric("Uptime", "99.9%")

    st.markdown("---")

    # Table statistics
    st.markdown("#### Table Statistics")

    tables = [
        {"name": "audit_findings", "records": "45,892", "size": "1.2 GB", "last_update": "2024-12-30 10:45"},
        {"name": "workpapers", "records": "128,456", "size": "2.1 GB", "last_update": "2024-12-30 10:42"},
        {"name": "documents", "records": "23,456", "size": "850 MB", "last_update": "2024-12-30 10:30"},
        {"name": "user_sessions", "records": "892,341", "size": "320 MB", "last_update": "2024-12-30 10:45"},
        {"name": "audit_logs", "records": "1,234,567", "size": "450 MB", "last_update": "2024-12-30 10:45"},
    ]

    for table in tables:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.5rem 0; border-bottom:1px solid {t['border']};">
            <span style="color:{t['text']}; font-weight:500; min-width:150px;">{table['name']}</span>
            <span style="color:{t['text_muted']};">{table['records']} records</span>
            <span style="color:{t['text_muted']};">{table['size']}</span>
            <span style="color:{t['text_muted']}; font-size:0.85rem;">{table['last_update']}</span>
        </div>
        """, unsafe_allow_html=True)

    # Maintenance actions
    st.markdown("---")
    st.markdown("#### Maintenance Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Run Backup", use_container_width=True):
            st.info("Backup initiated...")
    with col2:
        if st.button("🧹 Clean Old Logs", use_container_width=True):
            st.info("Cleaning logs older than 90 days...")
    with col3:
        if st.button("📊 Analyze Tables", use_container_width=True):
            st.info("Running table analysis...")


def _render_system_config(t: dict):
    """Render system configuration section."""
    st.markdown("### ⚙️ System Configuration")

    # General settings
    st.markdown("#### General Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("Application Name", value="AURIX Excellence 2026")
        st.text_input("Company Name", value="PT Bank Example")
        st.selectbox("Default Language", ["Bahasa Indonesia", "English"], index=0)

    with col2:
        st.text_input("Support Email", value="support@company.com")
        st.number_input("Session Timeout (minutes)", value=30, min_value=5, max_value=120)
        st.selectbox("Default Theme", ["Dark", "Light"], index=0)

    st.markdown("---")

    # Integration settings
    st.markdown("#### Integration Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("OJK API Endpoint", value="https://api.ojk.go.id/v1", type="password")
        st.text_input("BI SKNBI Endpoint", value="https://api.bi.go.id/sknbi", type="password")

    with col2:
        st.text_input("Email SMTP Server", value="smtp.company.com")
        st.text_input("Slack Webhook URL", value="https://hooks.slack.com/...", type="password")

    st.markdown("---")

    # Feature flags
    st.markdown("#### Feature Flags")

    col1, col2 = st.columns(2)

    with col1:
        st.checkbox("Enable AI Copilot", value=True)
        st.checkbox("Enable Executive Dashboard", value=True)
        st.checkbox("Enable Stress Testing", value=True)

    with col2:
        st.checkbox("Enable Process Mining", value=True)
        st.checkbox("Enable Gamification", value=True)
        st.checkbox("Enable Real-time Alerts", value=True)

    st.markdown("---")

    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        st.success("Configuration saved successfully!")
