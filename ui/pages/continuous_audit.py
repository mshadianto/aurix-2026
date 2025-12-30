"""
Continuous Audit Page Module for AURIX 2026.
Real-time monitoring, rule management, and alert dashboard.

Enhanced Features (World-Class 2026):
- AI-Powered AutoRuleGenerator (LLM-based rule creation)
- Natural Language rule definition
- AI-suggested rules from audit findings
- Rule effectiveness scoring and optimization
- ML-based false positive reduction
- Smart alert prioritization
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uuid
import random
import re

from ui.styles.css_builder import get_current_theme
from ui.components import (
    render_page_header,
    render_footer,
    render_badge,
    risk_badge,
    status_badge,
    render_metric_card,
    render_alert
)
from data.seeds import CONTINUOUS_AUDIT_RULES, AUDIT_UNIVERSE


class ContinuousAuditPage:
    """Continuous Audit monitoring and rule management page."""
    
    def __init__(self):
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state for continuous audit."""
        if 'ca_rules' not in st.session_state:
            # Load default rules from seeds
            st.session_state.ca_rules = [
                {**rule, 'enabled': True, 'alert_count': random.randint(0, 50)}
                for rule in CONTINUOUS_AUDIT_RULES
            ]
        
        if 'ca_alerts' not in st.session_state:
            st.session_state.ca_alerts = self._generate_sample_alerts()
    
    def _generate_sample_alerts(self) -> List[Dict]:
        """Generate sample alerts for demo."""
        alerts = []
        categories = ['AML', 'Financial', 'Security', 'Operations', 'Credit']
        severities = ['HIGH', 'MEDIUM', 'LOW']
        
        for i in range(15):
            rule = random.choice(CONTINUOUS_AUDIT_RULES)
            alerts.append({
                'id': f"ALT{i+1:04d}",
                'rule_id': rule['id'],
                'rule_name': rule['name'],
                'category': rule['category'],
                'severity': random.choice(severities),
                'description': f"Alert triggered: {rule['description']}",
                'timestamp': (datetime.now() - timedelta(hours=random.randint(1, 72))).strftime('%Y-%m-%d %H:%M'),
                'status': random.choice(['New', 'Reviewed', 'Escalated', 'Closed']),
                'details': {
                    'transaction_id': f"TXN{random.randint(100000, 999999)}",
                    'amount': random.randint(10000, 500000),
                    'account': f"ACC{random.randint(1000, 9999)}"
                }
            })
        
        return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def render(self):
        """Render the Continuous Audit page."""
        render_page_header("Continuous Audit", "Real-time monitoring and automated alerting")
        
        t = get_current_theme()
        
        # Summary Dashboard
        self._render_summary_dashboard()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tabs with enhanced AI features
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🚨 Alert Dashboard",
            "⚙️ Rule Management",
            "🤖 AI Rule Generator",
            "📊 Rule Effectiveness",
            "📈 Monitoring Stats",
            "➕ Create Rule"
        ])

        with tab1:
            self._render_alert_dashboard()

        with tab2:
            self._render_rule_management()

        with tab3:
            self._render_ai_rule_generator()

        with tab4:
            self._render_rule_effectiveness()

        with tab5:
            self._render_monitoring_stats()

        with tab6:
            self._render_create_rule()
        
        render_footer()
    
    def _render_summary_dashboard(self):
        """Render summary metrics."""
        t = get_current_theme()
        
        alerts = st.session_state.ca_alerts
        rules = st.session_state.ca_rules
        
        new_alerts = len([a for a in alerts if a['status'] == 'New'])
        high_alerts = len([a for a in alerts if a['severity'] == 'HIGH' and a['status'] != 'Closed'])
        active_rules = len([r for r in rules if r.get('enabled', True)])
        alerts_24h = len([a for a in alerts if 'hours' not in a['timestamp'] or int(a['timestamp'].split()[0]) <= 24])
        
        # Use Streamlit columns for metrics
        cols = st.columns(4)
        
        metrics_data = [
            ("ACTIVE RULES", str(active_rules), f"of {len(rules)} total", t['success']),
            ("NEW ALERTS", str(new_alerts), "Requires review", t['danger'] if new_alerts > 5 else t['success']),
            ("HIGH PRIORITY", str(high_alerts), "Immediate attention", t['danger'] if high_alerts > 0 else t['success']),
            ("TOTAL ALERTS (24H)", str(alerts_24h), "Last 24 hours", t['accent']),
        ]
        
        for col, (label, value, change, color) in zip(cols, metrics_data):
            with col:
                st.markdown(f'''
                <div style="background:{t['card']};border:1px solid {t['border']};border-radius:12px;padding:1.25rem;">
                    <div style="font-size:0.75rem;font-weight:500;text-transform:uppercase;letter-spacing:0.05em;color:{t['text_muted']} !important;margin-bottom:0.5rem;">{label}</div>
                    <div style="font-size:1.75rem;font-weight:700;color:{t['text']} !important;">{value}</div>
                    <div style="font-size:0.75rem;margin-top:0.25rem;color:{color} !important;">{change}</div>
                </div>
                ''', unsafe_allow_html=True)
    
    def _render_alert_dashboard(self):
        """Render the alert dashboard."""
        t = get_current_theme()
        
        st.markdown("### 🚨 Alert Dashboard")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_filter = st.selectbox(
                "Status",
                ["All", "New", "Reviewed", "Escalated", "Closed"],
                key="ca_alert_status"
            )
        
        with col2:
            severity_filter = st.selectbox(
                "Severity",
                ["All", "HIGH", "MEDIUM", "LOW"],
                key="ca_alert_severity"
            )
        
        with col3:
            category_filter = st.selectbox(
                "Category",
                ["All"] + list(set(a['category'] for a in st.session_state.ca_alerts)),
                key="ca_alert_category"
            )
        
        with col4:
            time_filter = st.selectbox(
                "Time Range",
                ["All", "Last 24 hours", "Last 7 days", "Last 30 days"],
                key="ca_alert_time"
            )
        
        st.markdown("---")
        
        # Apply filters
        filtered_alerts = st.session_state.ca_alerts.copy()
        
        if status_filter != "All":
            filtered_alerts = [a for a in filtered_alerts if a['status'] == status_filter]
        
        if severity_filter != "All":
            filtered_alerts = [a for a in filtered_alerts if a['severity'] == severity_filter]
        
        if category_filter != "All":
            filtered_alerts = [a for a in filtered_alerts if a['category'] == category_filter]
        
        # Display alerts
        if not filtered_alerts:
            st.info("✅ No alerts match the current filters.")
            return
        
        st.markdown(f"**Showing {len(filtered_alerts)} alert(s)**")
        
        for alert in filtered_alerts[:20]:
            self._render_alert_card(alert)
    
    def _render_alert_card(self, alert: Dict):
        """Render a single alert card."""
        t = get_current_theme()
        
        severity_colors = {
            'HIGH': t['danger'],
            'MEDIUM': t['warning'],
            'LOW': t['success']
        }
        
        status_colors = {
            'New': t['danger'],
            'Reviewed': t['warning'],
            'Escalated': t['accent'],
            'Closed': t['success']
        }
        
        sev_color = severity_colors.get(alert['severity'], t['text_muted'])
        stat_color = status_colors.get(alert['status'], t['text_muted'])
        
        with st.expander(f"🔔 {alert['id']}: {alert['rule_name']}", expanded=alert['status'] == 'New'):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f'''
                <div class="pro-card" style="padding:1rem;">
                    <div style="display:flex;gap:1rem;margin-bottom:1rem;">
                        <span class="badge" style="background:{sev_color}20;color:{sev_color};">{alert['severity']}</span>
                        <span class="badge" style="background:{stat_color}20;color:{stat_color};">{alert['status']}</span>
                        <span style="color:{t['text_muted']} !important;font-size:0.85rem;">Category: {alert['category']}</span>
                    </div>
                    <div style="color:{t['text']} !important;margin-bottom:1rem;">
                        {alert['description']}
                    </div>
                    <div style="background:{t['bg_secondary']};padding:1rem;border-radius:8px;font-family:monospace;font-size:0.85rem;">
                        <div style="color:{t['text_muted']} !important;margin-bottom:0.5rem;">Alert Details:</div>
                        <div style="color:{t['text']} !important;">Transaction ID: {alert['details']['transaction_id']}</div>
                        <div style="color:{t['text']} !important;">Amount: Rp {alert['details']['amount']:,}</div>
                        <div style="color:{t['text']} !important;">Account: {alert['details']['account']}</div>
                    </div>
                    <div style="margin-top:1rem;font-size:0.8rem;color:{t['text_muted']} !important;">
                        Timestamp: {alert['timestamp']}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Actions")
                
                new_status = st.selectbox(
                    "Update Status",
                    ["New", "Reviewed", "Escalated", "Closed"],
                    index=["New", "Reviewed", "Escalated", "Closed"].index(alert['status']),
                    key=f"alert_status_{alert['id']}"
                )
                
                if st.button("Update", key=f"update_alert_{alert['id']}"):
                    for a in st.session_state.ca_alerts:
                        if a['id'] == alert['id']:
                            a['status'] = new_status
                            break
                    st.success("✓ Status updated")
                    st.rerun()
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("🔍 Investigate", key=f"investigate_{alert['id']}"):
                    st.info("Opening investigation workflow...")
                
                if st.button("📋 Create Finding", key=f"create_finding_{alert['id']}"):
                    st.info("Redirecting to Findings module...")
    
    def _render_rule_management(self):
        """Render rule management interface."""
        t = get_current_theme()
        
        st.markdown("### ⚙️ Rule Management")
        
        # Filter by category
        categories = list(set(r['category'] for r in st.session_state.ca_rules))
        category_filter = st.selectbox(
            "Filter by Category",
            ["All"] + categories,
            key="rule_category_filter"
        )
        
        st.markdown("---")
        
        filtered_rules = st.session_state.ca_rules
        if category_filter != "All":
            filtered_rules = [r for r in filtered_rules if r['category'] == category_filter]
        
        # Display rules in a table-like format
        for rule in filtered_rules:
            enabled = rule.get('enabled', True)
            
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.markdown(f'''
                <div class="pro-card" style="padding:0.75rem;">
                    <div style="display:flex;align-items:center;gap:0.5rem;">
                        <span style="font-size:1.25rem;">{'🟢' if enabled else '🔴'}</span>
                        <div>
                            <div style="font-weight:600;color:{t['text']} !important;">{rule['name']}</div>
                            <div style="font-size:0.8rem;color:{t['text_secondary']} !important;">{rule['description']}</div>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                <div style="text-align:center;padding:0.5rem;">
                    <div style="font-size:0.75rem;color:{t['text_muted']} !important;">Category</div>
                    <div style="font-weight:600;color:{t['text']} !important;">{rule['category']}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'''
                <div style="text-align:center;padding:0.5rem;">
                    <div style="font-size:0.75rem;color:{t['text_muted']} !important;">Alerts</div>
                    <div style="font-weight:600;color:{t['warning'] if rule.get('alert_count', 0) > 10 else t['text']} !important;">
                        {rule.get('alert_count', 0)}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col4:
                if st.toggle("Enabled", value=enabled, key=f"rule_toggle_{rule['id']}"):
                    rule['enabled'] = True
                else:
                    rule['enabled'] = False
            
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
    
    def _render_monitoring_stats(self):
        """Render monitoring statistics."""
        t = get_current_theme()
        
        st.markdown("### 📈 Monitoring Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Alerts by Category
            st.markdown("#### Alerts by Category")
            
            alerts = st.session_state.ca_alerts
            category_counts = {}
            for a in alerts:
                cat = a['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            total = len(alerts)
            for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total * 100) if total > 0 else 0
                st.markdown(f'''
                <div class="pro-card" style="padding:0.75rem;margin-bottom:0.5rem;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
                        <span style="color:{t['text']} !important;">{cat}</span>
                        <span style="font-weight:600;color:{t['text']} !important;">{count} ({pct:.0f}%)</span>
                    </div>
                    <div class="progress-bar" style="height:4px;">
                        <div class="progress-fill" style="width:{pct}%;"></div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Alert Trend (mock data)
            st.markdown("#### Alert Trend (7 Days)")
            
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            values = [random.randint(5, 25) for _ in days]
            max_val = max(values)
            
            st.markdown(f'''
            <div class="pro-card" style="padding:1rem;">
                <div style="display:flex;align-items:end;gap:0.5rem;height:120px;">
            ''', unsafe_allow_html=True)
            
            bars_html = ""
            for i, (day, val) in enumerate(zip(days, values)):
                height = (val / max_val * 100) if max_val > 0 else 0
                bars_html += f'''
                <div style="flex:1;text-align:center;">
                    <div style="height:{height}px;background:{t['primary']};border-radius:4px 4px 0 0;"></div>
                    <div style="font-size:0.7rem;color:{t['text_muted']} !important;margin-top:4px;">{day}</div>
                    <div style="font-size:0.75rem;color:{t['text']} !important;">{val}</div>
                </div>
                '''
            
            st.markdown(f'''
                {bars_html}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            # Rule Performance
            st.markdown("#### Top Triggering Rules")
            
            rules = sorted(st.session_state.ca_rules, key=lambda x: x.get('alert_count', 0), reverse=True)[:5]
            
            for rule in rules:
                count = rule.get('alert_count', 0)
                st.markdown(f'''
                <div class="pro-card" style="padding:0.75rem;margin-bottom:0.5rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <div style="font-weight:600;color:{t['text']} !important;">{rule['name']}</div>
                            <div style="font-size:0.75rem;color:{t['text_muted']} !important;">{rule['category']}</div>
                        </div>
                        <div style="font-size:1.25rem;font-weight:700;color:{t['warning'] if count > 20 else t['text']} !important;">
                            {count}
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Response Time
            st.markdown("#### Alert Response Metrics")
            
            metrics = [
                ("Avg. Response Time", "2.5 hrs", t['success']),
                ("SLA Compliance", "94%", t['success']),
                ("False Positive Rate", "12%", t['warning']),
                ("Escalation Rate", "8%", t['text'])
            ]
            
            for name, value, color in metrics:
                st.markdown(f'''
                <div class="pro-card" style="padding:0.75rem;margin-bottom:0.5rem;">
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:{t['text_secondary']} !important;">{name}</span>
                        <span style="font-weight:600;color:{color} !important;">{value}</span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    
    def _render_create_rule(self):
        """Render create new rule form."""
        t = get_current_theme()
        
        st.markdown("### ➕ Create New Monitoring Rule")
        
        st.markdown(f'''
        <div class="pro-card" style="background:{t['bg_secondary']};margin-bottom:1rem;">
            <p style="color:{t['text_secondary']} !important;margin:0;">
                Define custom continuous audit rules to monitor specific patterns, thresholds, or anomalies in your data.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            rule_name = st.text_input(
                "Rule Name *",
                placeholder="e.g., High Value Wire Transfer",
                key="new_rule_name"
            )
            
            rule_description = st.text_area(
                "Description *",
                height=100,
                placeholder="Describe what this rule monitors...",
                key="new_rule_desc"
            )
            
            rule_category = st.selectbox(
                "Category *",
                ["AML", "Financial", "Security", "Operations", "Credit", "IT", "Fraud", "Compliance"],
                key="new_rule_category"
            )
        
        with col2:
            rule_type = st.selectbox(
                "Rule Type *",
                ["Threshold", "Pattern", "Anomaly", "Comparison", "Sequence"],
                key="new_rule_type"
            )
            
            if rule_type == "Threshold":
                threshold_value = st.number_input(
                    "Threshold Value",
                    min_value=0,
                    value=10000,
                    key="new_rule_threshold"
                )
                
                threshold_operator = st.selectbox(
                    "Operator",
                    ["Greater than", "Less than", "Equal to", "Not equal to"],
                    key="new_rule_operator"
                )
            
            severity = st.selectbox(
                "Default Severity",
                ["HIGH", "MEDIUM", "LOW"],
                index=1,
                key="new_rule_severity"
            )
            
            enabled = st.checkbox("Enable immediately", value=True, key="new_rule_enabled")
        
        st.markdown("---")
        
        # Data Source Configuration
        st.markdown("#### Data Source Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_source = st.selectbox(
                "Data Source",
                ["Transaction Database", "User Activity Logs", "System Logs", "External API", "Manual Input"],
                key="new_rule_source"
            )
            
            schedule = st.selectbox(
                "Monitoring Schedule",
                ["Real-time", "Every 5 minutes", "Hourly", "Daily", "Weekly"],
                key="new_rule_schedule"
            )
        
        with col2:
            notification = st.multiselect(
                "Notification Channels",
                ["Dashboard Alert", "Email", "SMS", "Slack", "Teams"],
                default=["Dashboard Alert"],
                key="new_rule_notification"
            )
            
            escalation = st.number_input(
                "Auto-escalate after (hours)",
                min_value=0,
                value=24,
                key="new_rule_escalation"
            )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("💾 Save Rule", type="primary", use_container_width=True):
                if not rule_name or not rule_description:
                    st.error("Please fill in all required fields.")
                else:
                    new_rule = {
                        'id': len(st.session_state.ca_rules) + 1,
                        'name': rule_name,
                        'description': rule_description,
                        'category': rule_category,
                        'threshold': threshold_value if rule_type == "Threshold" else None,
                        'enabled': enabled,
                        'alert_count': 0,
                        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M')
                    }
                    
                    st.session_state.ca_rules.append(new_rule)
                    st.success(f"✓ Rule '{rule_name}' created successfully!")
                    st.balloons()
        
        with col2:
            if st.button("🧪 Test Rule", use_container_width=True):
                st.info("Running rule test against sample data...")
                st.success("✓ Test completed: 3 matches found in sample data")

    def _render_ai_rule_generator(self):
        """Render AI-powered rule generator interface."""
        t = get_current_theme()

        st.markdown("### 🤖 AI Rule Generator")
        st.caption("Create monitoring rules using natural language or let AI suggest rules from patterns")

        # Natural Language Rule Creation
        st.markdown("#### 📝 Natural Language Rule Definition")
        st.markdown(f"""
        <div style="background:{t['bg_secondary']}; border:1px solid {t['border']}; border-radius:12px; padding:1rem; margin-bottom:1rem;">
            <p style="color:{t['text_muted']}; margin:0;">
                Describe the monitoring rule in plain English. The AI will convert it into a structured rule with appropriate thresholds and conditions.
            </p>
        </div>
        """, unsafe_allow_html=True)

        nl_input = st.text_area(
            "Describe your rule",
            placeholder="Example: Alert me when any single transaction exceeds 500 million rupiah, especially if it's a new customer or happens outside business hours",
            height=100,
            key="nl_rule_input"
        )

        if st.button("🔮 Generate Rule from Description", type="primary"):
            if nl_input:
                with st.spinner("AI analyzing description and generating rule..."):
                    # Simulate AI rule generation
                    generated_rule = self._generate_rule_from_nl(nl_input)

                    st.session_state['generated_rule'] = generated_rule
                    st.success("Rule generated successfully!")

        if 'generated_rule' in st.session_state:
            rule = st.session_state['generated_rule']

            st.markdown("#### Generated Rule")
            st.markdown(f"""
            <div style="background:{t['card']}; border:1px solid {t['primary']}; border-radius:12px; padding:1.5rem; margin:1rem 0;">
                <div style="font-weight:700; color:{t['text']}; font-size:1.1rem; margin-bottom:0.75rem;">
                    {rule['name']}
                </div>
                <div style="color:{t['text_muted']}; margin-bottom:1rem;">
                    {rule['description']}
                </div>
                <div style="display:flex; gap:1rem; flex-wrap:wrap;">
                    <div style="background:{t['bg_secondary']}; padding:0.5rem 1rem; border-radius:8px;">
                        <span style="color:{t['text_muted']}; font-size:0.8rem;">Category:</span>
                        <span style="color:{t['text']}; font-weight:600;"> {rule['category']}</span>
                    </div>
                    <div style="background:{t['bg_secondary']}; padding:0.5rem 1rem; border-radius:8px;">
                        <span style="color:{t['text_muted']}; font-size:0.8rem;">Severity:</span>
                        <span style="color:{t['warning']}; font-weight:600;"> {rule['severity']}</span>
                    </div>
                    <div style="background:{t['bg_secondary']}; padding:0.5rem 1rem; border-radius:8px;">
                        <span style="color:{t['text_muted']}; font-size:0.8rem;">Threshold:</span>
                        <span style="color:{t['text']}; font-weight:600;"> {rule['threshold']}</span>
                    </div>
                </div>
                <div style="margin-top:1rem; padding-top:1rem; border-top:1px solid {t['border']};">
                    <div style="font-size:0.85rem; color:{t['text_muted']};">Conditions:</div>
                    <ul style="color:{t['text']}; margin-top:0.5rem;">
            """, unsafe_allow_html=True)

            for condition in rule.get('conditions', []):
                text_color = t['text']
                st.markdown(f"<li style='color:{text_color};'>{condition}</li>", unsafe_allow_html=True)

            st.markdown("</ul></div></div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Accept & Create Rule", use_container_width=True):
                    rule['enabled'] = True
                    rule['alert_count'] = 0
                    rule['id'] = len(st.session_state.ca_rules) + 100
                    st.session_state.ca_rules.append(rule)
                    del st.session_state['generated_rule']
                    st.success("Rule created and activated!")
                    st.rerun()
            with col2:
                if st.button("✏️ Edit Before Creating", use_container_width=True):
                    st.info("Opening rule editor...")

        # AI-Suggested Rules
        st.markdown("---")
        st.markdown("#### 💡 AI-Suggested Rules")
        st.caption("Based on analysis of recent alerts, findings, and industry patterns")

        if st.button("🔍 Analyze & Suggest New Rules"):
            with st.spinner("AI analyzing patterns and generating suggestions..."):
                suggestions = self._generate_ai_suggestions()
                st.session_state['ai_suggestions'] = suggestions

        if 'ai_suggestions' in st.session_state:
            for i, suggestion in enumerate(st.session_state['ai_suggestions']):
                confidence_color = t['success'] if suggestion['confidence'] >= 0.8 else t['warning'] if suggestion['confidence'] >= 0.6 else t['text_muted']

                st.markdown(f"""
                <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:12px; padding:1rem; margin:0.75rem 0;">
                    <div style="display:flex; justify-content:space-between; align-items:start;">
                        <div style="flex:1;">
                            <div style="font-weight:700; color:{t['text']};">{suggestion['name']}</div>
                            <div style="color:{t['text_muted']}; font-size:0.9rem; margin:0.5rem 0;">
                                {suggestion['description']}
                            </div>
                            <div style="font-size:0.85rem; color:{t['text_muted']};">
                                Based on: <span style="color:{t['accent']};">{suggestion['basis']}</span>
                            </div>
                        </div>
                        <div style="text-align:center; padding:0 1rem;">
                            <div style="font-size:1.5rem; font-weight:700; color:{confidence_color};">
                                {suggestion['confidence']*100:.0f}%
                            </div>
                            <div style="font-size:0.7rem; color:{t['text_muted']};">confidence</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"✅ Accept", key=f"accept_sug_{i}"):
                        suggestion['enabled'] = True
                        suggestion['alert_count'] = 0
                        suggestion['id'] = len(st.session_state.ca_rules) + 200 + i
                        st.session_state.ca_rules.append(suggestion)
                        st.success(f"Rule '{suggestion['name']}' created!")
                with col2:
                    if st.button(f"👎 Dismiss", key=f"dismiss_sug_{i}"):
                        st.info("Suggestion dismissed")
                with col3:
                    if st.button(f"🔄 Modify", key=f"modify_sug_{i}"):
                        st.info("Opening editor...")

    def _render_rule_effectiveness(self):
        """Render rule effectiveness scoring and optimization."""
        t = get_current_theme()

        st.markdown("### 📊 Rule Effectiveness Analysis")
        st.caption("ML-based analysis of rule performance and false positive reduction")

        # Overall effectiveness metrics
        st.markdown("#### Overall Performance")

        ecol1, ecol2, ecol3, ecol4 = st.columns(4)

        with ecol1:
            st.markdown(f"""
            <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:12px; padding:1rem; text-align:center;">
                <div style="font-size:2rem; font-weight:700; color:{t['success']};">87%</div>
                <div style="font-size:0.8rem; color:{t['text_muted']};">True Positive Rate</div>
            </div>
            """, unsafe_allow_html=True)

        with ecol2:
            st.markdown(f"""
            <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:12px; padding:1rem; text-align:center;">
                <div style="font-size:2rem; font-weight:700; color:{t['warning']};">13%</div>
                <div style="font-size:0.8rem; color:{t['text_muted']};">False Positive Rate</div>
            </div>
            """, unsafe_allow_html=True)

        with ecol3:
            st.markdown(f"""
            <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:12px; padding:1rem; text-align:center;">
                <div style="font-size:2rem; font-weight:700; color:{t['accent']};">2.3h</div>
                <div style="font-size:0.8rem; color:{t['text_muted']};">Avg Response Time</div>
            </div>
            """, unsafe_allow_html=True)

        with ecol4:
            st.markdown(f"""
            <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:12px; padding:1rem; text-align:center;">
                <div style="font-size:2rem; font-weight:700; color:{t['text']};">94%</div>
                <div style="font-size:0.8rem; color:{t['text_muted']};">SLA Compliance</div>
            </div>
            """, unsafe_allow_html=True)

        # Individual rule effectiveness
        st.markdown("---")
        st.markdown("#### Rule Performance Ranking")

        rule_performance = self._calculate_rule_effectiveness()

        for i, perf in enumerate(rule_performance[:10]):
            eff_score = perf['effectiveness']
            eff_color = t['success'] if eff_score >= 80 else t['warning'] if eff_score >= 60 else t['danger']
            fp_color = t['success'] if perf['false_positive_rate'] < 10 else t['warning'] if perf['false_positive_rate'] < 20 else t['danger']

            st.markdown(f"""
            <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:12px; padding:1rem; margin:0.5rem 0;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="flex:2;">
                        <div style="font-weight:600; color:{t['text']};">{perf['name']}</div>
                        <div style="font-size:0.8rem; color:{t['text_muted']};">{perf['category']}</div>
                    </div>
                    <div style="flex:1; text-align:center;">
                        <div style="font-size:1.25rem; font-weight:700; color:{eff_color};">{eff_score}%</div>
                        <div style="font-size:0.7rem; color:{t['text_muted']};">Effectiveness</div>
                    </div>
                    <div style="flex:1; text-align:center;">
                        <div style="font-size:1.25rem; font-weight:600; color:{fp_color};">{perf['false_positive_rate']}%</div>
                        <div style="font-size:0.7rem; color:{t['text_muted']};">False Positive</div>
                    </div>
                    <div style="flex:1; text-align:center;">
                        <div style="font-size:1.25rem; font-weight:600; color:{t['text']};">{perf['alerts']}</div>
                        <div style="font-size:0.7rem; color:{t['text_muted']};">Total Alerts</div>
                    </div>
                    <div style="flex:1; text-align:right;">
            """, unsafe_allow_html=True)

            if perf['recommendation']:
                rec_color = t['warning'] if 'adjust' in perf['recommendation'].lower() else t['danger'] if 'disable' in perf['recommendation'].lower() else t['success']
                st.markdown(f"""
                        <span style="background:{rec_color}20; color:{rec_color}; padding:0.25rem 0.5rem; border-radius:4px; font-size:0.75rem;">
                            {perf['recommendation']}
                        </span>
                """, unsafe_allow_html=True)

            st.markdown("</div></div></div>", unsafe_allow_html=True)

        # ML-based false positive reduction
        st.markdown("---")
        st.markdown("#### 🧠 ML-Based False Positive Reduction")
        st.caption("AI continuously learns from analyst feedback to reduce false positives")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg, {t['success']}10 0%, {t['accent']}10 100%);
                    border:1px solid {t['success']}40; border-radius:12px; padding:1.5rem; margin:1rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-weight:700; color:{t['text']}; font-size:1.1rem;">
                        ML Model Performance
                    </div>
                    <div style="color:{t['text_muted']}; margin-top:0.5rem;">
                        Model has been trained on 12,450 analyst decisions
                    </div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:2rem; font-weight:700; color:{t['success']};">-34%</div>
                    <div style="font-size:0.8rem; color:{t['text_muted']};">False Positives Reduced</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Optimization suggestions
        st.markdown("#### 🎯 Optimization Recommendations")

        optimizations = [
            {
                'rule': 'High Value Transfer',
                'current_threshold': 'Rp 500,000,000',
                'suggested_threshold': 'Rp 750,000,000',
                'impact': 'Reduce FP by 28%',
                'reason': 'Analysis shows 95% of legitimate transactions are below Rp 750M'
            },
            {
                'rule': 'Rapid Successive Transactions',
                'current_threshold': '3 transactions in 10 minutes',
                'suggested_threshold': '5 transactions in 10 minutes',
                'impact': 'Reduce FP by 42%',
                'reason': 'Normal business customers often make 3-4 quick transactions'
            },
            {
                'rule': 'Dormant Account Activity',
                'current_threshold': '90 days inactive',
                'suggested_threshold': '180 days inactive',
                'impact': 'Reduce FP by 55%',
                'reason': 'Many accounts have seasonal activity patterns'
            }
        ]

        for opt in optimizations:
            st.markdown(f"""
            <div style="background:{t['card']}; border:1px solid {t['border']}; border-left:4px solid {t['accent']};
                        border-radius:0 12px 12px 0; padding:1rem; margin:0.75rem 0;">
                <div style="font-weight:700; color:{t['text']};">{opt['rule']}</div>
                <div style="display:flex; gap:2rem; margin:0.75rem 0;">
                    <div>
                        <span style="color:{t['text_muted']}; font-size:0.85rem;">Current: </span>
                        <span style="color:{t['danger']};">{opt['current_threshold']}</span>
                    </div>
                    <span style="color:{t['text_muted']};">→</span>
                    <div>
                        <span style="color:{t['text_muted']}; font-size:0.85rem;">Suggested: </span>
                        <span style="color:{t['success']}; font-weight:600;">{opt['suggested_threshold']}</span>
                    </div>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:0.85rem; color:{t['text_muted']};">{opt['reason']}</div>
                    <div style="background:{t['success']}; color:white; padding:0.25rem 0.75rem; border-radius:4px; font-weight:600; font-size:0.8rem;">
                        {opt['impact']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ Apply Suggestion", key=f"apply_opt_{opt['rule']}"):
                    st.success(f"Threshold updated for {opt['rule']}")
            with col2:
                if st.button(f"❌ Dismiss", key=f"dismiss_opt_{opt['rule']}"):
                    st.info("Suggestion dismissed")

    def _generate_rule_from_nl(self, description: str) -> Dict:
        """Generate rule from natural language description (AI simulation)."""
        # Simulate AI rule generation
        import time
        time.sleep(1)  # Simulate processing

        # Extract key elements from description
        has_amount = 'million' in description.lower() or 'rupiah' in description.lower() or 'rp' in description.lower()
        has_time = 'hour' in description.lower() or 'time' in description.lower() or 'business' in description.lower()
        has_customer = 'customer' in description.lower() or 'new' in description.lower()

        conditions = []
        if has_amount:
            conditions.append("Transaction amount > Rp 500,000,000")
        if has_time:
            conditions.append("Transaction time outside 08:00-17:00 WIB")
        if has_customer:
            conditions.append("Customer account age < 30 days")

        return {
            'name': 'High Value Transaction - Extended Criteria',
            'description': 'Monitor high-value transactions with additional risk factors including new customers and off-hours activity',
            'category': 'AML',
            'severity': 'HIGH',
            'threshold': 'Rp 500,000,000',
            'conditions': conditions if conditions else [
                "Transaction amount exceeds defined threshold",
                "Additional risk factors present"
            ]
        }

    def _generate_ai_suggestions(self) -> List[Dict]:
        """Generate AI-suggested rules based on pattern analysis."""
        import time
        time.sleep(1.5)

        return [
            {
                'name': 'Structuring Detection - Enhanced',
                'description': 'Detect potential structuring behavior where transactions are split to avoid reporting thresholds',
                'category': 'AML',
                'severity': 'HIGH',
                'basis': 'Pattern analysis of 847 flagged transactions',
                'confidence': 0.92
            },
            {
                'name': 'Velocity Anomaly - Business Hours',
                'description': 'Flag accounts with unusual transaction velocity during business hours compared to their historical baseline',
                'category': 'Fraud',
                'severity': 'MEDIUM',
                'basis': 'ML model trained on confirmed fraud cases',
                'confidence': 0.78
            },
            {
                'name': 'Geographic Risk - High-Risk Jurisdiction',
                'description': 'Monitor transactions involving counterparties in FATF-identified high-risk jurisdictions',
                'category': 'AML',
                'severity': 'HIGH',
                'basis': 'Updated FATF grey list and industry advisories',
                'confidence': 0.95
            },
            {
                'name': 'Account Takeover Pattern',
                'description': 'Detect potential account takeover based on login location, device change, and immediate high-value transfer',
                'category': 'Security',
                'severity': 'HIGH',
                'basis': 'Security incident correlation analysis',
                'confidence': 0.85
            }
        ]

    def _calculate_rule_effectiveness(self) -> List[Dict]:
        """Calculate effectiveness scores for each rule."""
        performances = []

        for rule in st.session_state.ca_rules:
            alerts = rule.get('alert_count', 0)

            # Simulate effectiveness metrics
            effectiveness = random.randint(55, 98)
            fp_rate = random.randint(3, 25)

            # Generate recommendation based on performance
            if effectiveness < 60:
                recommendation = "Consider disabling"
            elif fp_rate > 20:
                recommendation = "Adjust threshold"
            elif effectiveness > 90 and fp_rate < 10:
                recommendation = "Optimal"
            else:
                recommendation = "Monitor"

            performances.append({
                'name': rule['name'],
                'category': rule['category'],
                'alerts': alerts,
                'effectiveness': effectiveness,
                'false_positive_rate': fp_rate,
                'recommendation': recommendation
            })

        return sorted(performances, key=lambda x: x['effectiveness'], reverse=True)


def render():
    """Entry point for the Continuous Audit page."""
    page = ContinuousAuditPage()
    page.render()
