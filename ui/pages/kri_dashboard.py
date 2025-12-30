"""
KRI Dashboard Page Module for AURIX 2026.
Key Risk Indicators monitoring and visualization.

Enhanced Features (World-Class 2026):
- Active KRI Cards with AI Analysis triggers
- Early Warning Indicators (predictive breach detection)
- Trend-adjusted dynamic thresholds
- KRI correlation matrix analysis
- Cascading risk visualization
- Automated escalation workflow
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
import numpy as np

from ui.styles.css_builder import get_current_theme
from ui.components import (
    render_page_header,
    render_footer,
    render_badge,
    render_metric_card,
    render_kri_gauge,
    render_progress_bar
)
from ui.components.active_kri_card import render_active_kri_card, KRIStatus
from data.seeds import KRI_INDICATORS


class KRIDashboardPage:
    """KRI Dashboard with interactive visualizations."""
    
    def __init__(self):
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state for KRI data."""
        if 'kri_values' not in st.session_state:
            st.session_state.kri_values = self._generate_sample_kri_data()
        
        if 'kri_history' not in st.session_state:
            st.session_state.kri_history = self._generate_historical_data()
    
    def _generate_sample_kri_data(self) -> Dict:
        """Generate sample KRI values for demo."""
        kri_data = {}
        
        for category, indicators in KRI_INDICATORS.items():
            kri_data[category] = {}
            for ind in indicators:
                name = ind['name']
                threshold = ind['threshold']
                
                # Generate realistic values based on threshold
                if threshold == 0:
                    value = random.randint(0, 3)
                elif ind['good_direction'] == 'lower':
                    value = threshold * random.uniform(0.5, 1.2)
                elif ind['good_direction'] == 'higher':
                    value = threshold * random.uniform(0.85, 1.15)
                else:
                    value = threshold * random.uniform(0.9, 1.1)
                
                kri_data[category][name] = {
                    'value': round(value, 2),
                    'threshold': threshold,
                    'unit': ind['unit'],
                    'good_direction': ind['good_direction'],
                    'trend': random.choice(['up', 'down', 'stable']),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
                }
        
        return kri_data
    
    def _generate_historical_data(self) -> Dict:
        """Generate historical KRI data for trends."""
        history = {}
        
        for category, indicators in KRI_INDICATORS.items():
            history[category] = {}
            for ind in indicators:
                name = ind['name']
                threshold = ind['threshold']
                
                # Generate 12 months of data
                values = []
                base_value = threshold * 0.8 if threshold > 0 else 1
                
                for i in range(12):
                    variation = random.uniform(-0.15, 0.15)
                    val = base_value * (1 + variation)
                    values.append(round(val, 2))
                    base_value = val
                
                history[category][name] = values
        
        return history
    
    def render(self):
        """Render the KRI Dashboard page."""
        render_page_header("KRI Dashboard", "Key Risk Indicators Monitoring & Analysis")

        t = get_current_theme()

        # Overall Risk Summary
        self._render_risk_summary()

        st.markdown("<br>", unsafe_allow_html=True)

        # Main tabs for enhanced features
        main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
            "📊 KRI Monitoring",
            "⚡ Early Warning",
            "🔗 Risk Correlation",
            "📋 Escalation"
        ])

        with main_tab1:
            # Category selector
            categories = list(KRI_INDICATORS.keys())
            tab_names = [f"📊 {cat}" for cat in categories]
            tabs = st.tabs(tab_names)

            for tab, category in zip(tabs, categories):
                with tab:
                    self._render_category_dashboard(category)

        with main_tab2:
            self._render_early_warning_tab(t)

        with main_tab3:
            self._render_correlation_tab(t)

        with main_tab4:
            self._render_escalation_tab(t)

        render_footer()
    
    def _render_risk_summary(self):
        """Render overall risk summary."""
        t = get_current_theme()
        
        # Calculate breach counts
        total_kris = 0
        breached = 0
        warning = 0
        
        for category, indicators in st.session_state.kri_values.items():
            for name, data in indicators.items():
                total_kris += 1
                value = data['value']
                threshold = data['threshold']
                direction = data['good_direction']
                
                if threshold == 0:
                    if value > 0:
                        breached += 1
                elif direction == 'lower':
                    if value > threshold:
                        breached += 1
                    elif value > threshold * 0.8:
                        warning += 1
                elif direction == 'higher':
                    if value < threshold:
                        breached += 1
                    elif value < threshold * 1.1:
                        warning += 1
        
        healthy = total_kris - breached - warning
        
        # Use Streamlit columns for metrics
        cols = st.columns(4)
        
        metrics_data = [
            ("TOTAL KRIS", str(total_kris), "Monitored indicators", t['accent']),
            ("HEALTHY", str(healthy), f"{healthy/total_kris*100:.0f}% of total", t['success']),
            ("WARNING", str(warning), "Near threshold", t['warning']),
            ("BREACHED", str(breached), "Action required", t['danger'] if breached > 0 else t['success']),
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
        
        # Risk heat indicator
        if breached > 2:
            risk_level = "HIGH"
            risk_color = t['danger']
        elif breached > 0 or warning > 3:
            risk_level = "MEDIUM"
            risk_color = t['warning']
        else:
            risk_level = "LOW"
            risk_color = t['success']
        
        st.markdown(f'''
        <div style="text-align:center;padding:1rem;background:linear-gradient(135deg, {risk_color}15, {risk_color}05);border:1px solid {risk_color}30;border-radius:12px;margin-top:1rem;">
            <div style="font-size:0.85rem;color:{t['text_muted']} !important;">Overall Risk Level</div>
            <div style="font-size:2rem;font-weight:700;color:{risk_color} !important;">{risk_level}</div>
            <div style="font-size:0.8rem;color:{t['text_secondary']} !important;">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    def _render_category_dashboard(self, category: str):
        """Render dashboard for a specific KRI category."""
        t = get_current_theme()
        
        st.markdown(f"### {category}")
        
        indicators = st.session_state.kri_values.get(category, {})
        
        if not indicators:
            st.info("No indicators configured for this category.")
            return
        
        # Create gauge grid
        cols = st.columns(3)
        
        for i, (name, data) in enumerate(indicators.items()):
            with cols[i % 3]:
                self._render_kri_card(name, data, category)
        
        # Trend Analysis
        st.markdown("---")
        st.markdown("#### 📈 Trend Analysis (12 Months)")
        
        history = st.session_state.kri_history.get(category, {})
        
        selected_kri = st.selectbox(
            "Select KRI for trend view",
            options=list(indicators.keys()),
            key=f"kri_trend_{category}"
        )
        
        if selected_kri and selected_kri in history:
            self._render_trend_chart(selected_kri, history[selected_kri], indicators[selected_kri])
    
    def _render_kri_card(self, name: str, data: Dict, category: str):
        """Render a single KRI card with AI analysis trigger (2026 Enhancement)."""
        value = data['value']
        threshold = data['threshold']
        unit = data['unit']
        direction = data['good_direction']
        trend = data['trend']
        
        # Map metric name to ID for analysis
        metric_id = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')
        
        # Determine if lower is worse
        lower_is_worse = direction == 'higher'
        
        # Get trend value (mock)
        trend_value = round(value * 0.05, 2) if trend != 'stable' else None
        trend_direction = trend if trend != 'stable' else None
        
        # Use the new Active KRI Card component (2026)
        render_active_kri_card(
            metric_id=metric_id,
            label=name,
            value=value,
            threshold=threshold,
            unit=unit,
            trend_value=trend_value,
            trend_direction=trend_direction,
            lower_is_worse=lower_is_worse
        )
    
    def _render_trend_chart(self, name: str, values: List[float], current_data: Dict):
        """Render a simple trend chart."""
        t = get_current_theme()
        
        threshold = current_data['threshold']
        unit = current_data['unit']
        
        # Generate month labels
        months = []
        for i in range(11, -1, -1):
            month = (datetime.now() - timedelta(days=30*i)).strftime('%b')
            months.append(month)
        
        # Calculate chart dimensions
        max_val = max(max(values), threshold * 1.2 if threshold > 0 else max(values) * 1.2)
        min_val = min(min(values), threshold * 0.5 if threshold > 0 else 0)
        range_val = max_val - min_val if max_val > min_val else 1
        
        chart_height = 200
        
        # Header
        st.markdown(f'''
        <div class="pro-card" style="padding:1rem;">
            <div style="display:flex;justify-content:space-between;margin-bottom:1rem;">
                <span style="font-weight:600;color:{t['text']} !important;">{name} - 12 Month Trend</span>
                <span style="color:{t['text_muted']} !important;">Threshold: {threshold}{unit}</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Use Streamlit columns for the chart
        cols = st.columns(12)
        for i, (col, month, val) in enumerate(zip(cols, months, values)):
            with col:
                height_pct = ((val - min_val) / range_val * 100) if range_val > 0 else 50
                color = t['success'] if val <= threshold or threshold == 0 else t['danger']
                st.markdown(f'''
                <div style="display:flex;flex-direction:column;align-items:center;height:{chart_height}px;">
                    <div style="flex:1;width:100%;display:flex;align-items:end;">
                        <div style="width:100%;height:{height_pct}%;background:{color};border-radius:4px 4px 0 0;min-height:4px;"></div>
                    </div>
                    <div style="font-size:0.6rem;color:{t['text_muted']} !important;margin-top:4px;">{month}</div>
                    <div style="font-size:0.65rem;color:{t['text']} !important;">{val:.1f}</div>
                </div>
                ''', unsafe_allow_html=True)
        
        # Stats footer
        st.markdown(f'''
        <div class="pro-card" style="padding:1rem;margin-top:0.5rem;">
            <div style="display:flex;justify-content:space-around;font-size:0.85rem;">
                <div>
                    <span style="color:{t['text_muted']} !important;">Min:</span>
                    <span style="font-weight:600;color:{t['text']} !important;"> {min(values):.2f}{unit}</span>
                </div>
                <div>
                    <span style="color:{t['text_muted']} !important;">Max:</span>
                    <span style="font-weight:600;color:{t['text']} !important;"> {max(values):.2f}{unit}</span>
                </div>
                <div>
                    <span style="color:{t['text_muted']} !important;">Avg:</span>
                    <span style="font-weight:600;color:{t['text']} !important;"> {sum(values)/len(values):.2f}{unit}</span>
                </div>
                <div>
                    <span style="color:{t['text_muted']} !important;">Current:</span>
                    <span style="font-weight:600;color:{t['text']} !important;"> {current_data['value']}{unit}</span>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Update KRI value
        col1, col2 = st.columns([3, 1])
        
        with col1:
            new_value = st.number_input(
                f"Update {name} value",
                value=float(current_data['value']),
                step=0.1,
                key=f"update_kri_{name}"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📥 Update", key=f"btn_update_{name}"):
                # Find and update the value
                for cat, indicators in st.session_state.kri_values.items():
                    if name in indicators:
                        indicators[name]['value'] = new_value
                        indicators[name]['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                        st.success(f"✓ {name} updated to {new_value}{unit}")
                        st.rerun()


    def _render_early_warning_tab(self, t: dict):
        """Render Early Warning Indicators tab."""
        st.markdown("### ⚡ Early Warning System")
        st.caption("AI-powered predictive breach detection with trend-adjusted thresholds")

        # Generate early warnings
        warnings = self._calculate_early_warnings()

        if warnings:
            # Alert summary
            critical_count = sum(1 for w in warnings if w['severity'] == 'critical')
            warning_count = sum(1 for w in warnings if w['severity'] == 'warning')

            st.markdown(f"""
            <div style="background:linear-gradient(135deg, {t['warning']}15 0%, {t['danger']}10 100%);
                        border:2px solid {t['warning']}50; border-radius:16px; padding:1.5rem; margin:1rem 0;">
                <div style="display:flex; align-items:center; gap:1rem;">
                    <span style="font-size:2.5rem;">⚡</span>
                    <div>
                        <div style="font-size:1.25rem; font-weight:700; color:{t['text']};">
                            {len(warnings)} Early Warning Signal{'s' if len(warnings) > 1 else ''}
                        </div>
                        <div style="color:{t['text_muted']};">
                            {critical_count} critical, {warning_count} warning - action recommended within 30 days
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Individual warnings
            for warning in warnings:
                severity_color = t['danger'] if warning['severity'] == 'critical' else t['warning']
                trend_icon = "📈" if warning['trend'] == 'up' else "📉" if warning['trend'] == 'down' else "➡️"

                st.markdown(f"""
                <div style="background:{t['card']}; border:1px solid {severity_color}; border-left:4px solid {severity_color};
                            border-radius:0 12px 12px 0; padding:1rem; margin:0.75rem 0;">
                    <div style="display:flex; justify-content:space-between; align-items:start;">
                        <div style="flex:1;">
                            <div style="font-weight:700; color:{t['text']}; font-size:1.1rem;">
                                {trend_icon} {warning['kri_name']}
                            </div>
                            <div style="color:{t['text_muted']}; margin:0.5rem 0;">
                                Current: <strong>{warning['current_value']:.2f}{warning['unit']}</strong> |
                                Threshold: <strong>{warning['threshold']:.2f}{warning['unit']}</strong>
                            </div>
                            <div style="color:{t['text_muted']}; font-size:0.85rem;">
                                {warning['message']}
                            </div>
                        </div>
                        <div style="text-align:center; padding:0 1rem;">
                            <div style="font-size:1.75rem; font-weight:800; color:{severity_color};">
                                {warning['days_to_breach']}
                            </div>
                            <div style="font-size:0.7rem; color:{t['text_muted']};">days to breach</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="background:{severity_color}; color:white; padding:0.4rem 0.8rem;
                                        border-radius:8px; font-weight:600; font-size:0.8rem;">
                                {warning['severity'].upper()}
                            </div>
                            <div style="font-size:0.75rem; color:{t['text_muted']}; margin-top:0.5rem;">
                                Velocity: {warning['velocity']:+.2f}/day
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Mitigation actions
                with st.expander(f"📋 Recommended Actions for {warning['kri_name']}"):
                    for i, action in enumerate(warning['actions'], 1):
                        st.markdown(f"{i}. {action}")
        else:
            st.success("✅ No early warning signals detected. All KRIs trending safely within thresholds.")

        # Trend-adjusted thresholds
        st.markdown("---")
        st.markdown("### 📊 Dynamic Threshold Analysis")
        st.caption("Thresholds adjusted based on trend velocity and seasonality")

        # Show adjustments table
        adjustments = self._calculate_threshold_adjustments()

        for adj in adjustments[:8]:
            pct_change = (adj['adjusted'] - adj['original']) / adj['original'] * 100 if adj['original'] != 0 else 0
            change_color = t['warning'] if abs(pct_change) > 5 else t['text_muted']

            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:0.5rem; border-bottom:1px solid {t['border']};">
                <span style="color:{t['text']}; font-weight:500; flex:2;">{adj['kri_name']}</span>
                <span style="color:{t['text_muted']}; flex:1; text-align:center;">Original: {adj['original']:.2f}</span>
                <span style="color:{t['accent']}; flex:1; text-align:center; font-weight:600;">Adjusted: {adj['adjusted']:.2f}</span>
                <span style="color:{change_color}; flex:1; text-align:right;">{pct_change:+.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    def _render_correlation_tab(self, t: dict):
        """Render KRI Correlation Matrix tab."""
        st.markdown("### 🔗 KRI Correlation Analysis")
        st.caption("Identify interconnected risks and potential cascade effects")

        # Get all KRI names for correlation
        kri_names = []
        kri_values = []

        for category, indicators in st.session_state.kri_values.items():
            for name, data in indicators.items():
                kri_names.append(name[:15])  # Truncate for display
                kri_values.append(data['value'])

        # Generate correlation matrix (using historical data)
        n = len(kri_names)
        if n > 1:
            # Create mock correlation matrix
            corr_matrix = np.eye(n)
            for i in range(n):
                for j in range(i + 1, n):
                    corr = random.uniform(-0.5, 0.9)
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

            # Display as heatmap (text-based for Streamlit)
            st.markdown("#### Correlation Heatmap")

            # Find high correlations
            high_corr = []
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(corr_matrix[i, j]) > 0.6:
                        high_corr.append((kri_names[i], kri_names[j], corr_matrix[i, j]))

            if high_corr:
                st.markdown("**Significant Correlations Detected:**")
                for kri1, kri2, corr in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:5]:
                    corr_type = "positive" if corr > 0 else "negative"
                    color = t['danger'] if corr > 0.7 else t['warning'] if corr > 0 else t['info']

                    st.markdown(f"""
                    <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:8px;
                                padding:0.75rem; margin:0.5rem 0;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span style="font-weight:600; color:{t['text']};">{kri1}</span>
                                <span style="color:{t['text_muted']};"> ↔ </span>
                                <span style="font-weight:600; color:{t['text']};">{kri2}</span>
                            </div>
                            <div style="background:{color}; color:white; padding:0.25rem 0.75rem;
                                        border-radius:4px; font-weight:600;">
                                {corr:.2f} ({corr_type})
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Cascading Risk Visualization
            st.markdown("---")
            st.markdown("### ⛓️ Cascading Risk Analysis")
            st.caption("Visualize how one KRI breach can trigger others")

            # Select source KRI
            source_kri = st.selectbox("Select Source KRI", kri_names)

            if source_kri:
                # Find correlated KRIs
                source_idx = kri_names.index(source_kri)
                cascades = []

                for i, name in enumerate(kri_names):
                    if i != source_idx:
                        corr = corr_matrix[source_idx, i]
                        if abs(corr) > 0.4:
                            impact = "High" if abs(corr) > 0.7 else "Medium"
                            direction = "same" if corr > 0 else "opposite"
                            cascades.append({
                                'name': name,
                                'correlation': corr,
                                'impact': impact,
                                'direction': direction
                            })

                if cascades:
                    st.markdown(f"**Cascade from {source_kri}:**")

                    for cascade in sorted(cascades, key=lambda x: abs(x['correlation']), reverse=True):
                        impact_color = t['danger'] if cascade['impact'] == 'High' else t['warning']
                        arrow = "↗" if cascade['direction'] == 'same' else "↘"

                        st.markdown(f"""
                        <div style="display:flex; align-items:center; padding:0.5rem 0; border-bottom:1px solid {t['border']};">
                            <span style="font-size:1.5rem; margin-right:0.75rem;">{arrow}</span>
                            <span style="flex:1; color:{t['text']};">{cascade['name']}</span>
                            <span style="background:{impact_color}20; color:{impact_color}; padding:0.2rem 0.5rem;
                                        border-radius:4px; font-size:0.8rem; font-weight:600;">
                                {cascade['impact']} Impact
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(f"No significant cascade relationships found for {source_kri}")

    def _render_escalation_tab(self, t: dict):
        """Render Automated Escalation Workflow tab."""
        st.markdown("### 📋 Escalation Workflow")
        st.caption("Automated escalation based on severity and breach duration")

        # Current escalations
        escalations = self._get_current_escalations()

        if escalations:
            st.markdown(f"""
            <div style="background:{t['danger']}10; border:1px solid {t['danger']}40; border-radius:12px;
                        padding:1rem; margin:1rem 0;">
                <div style="font-weight:700; color:{t['danger']}; font-size:1.1rem;">
                    {len(escalations)} Active Escalation{'s' if len(escalations) > 1 else ''}
                </div>
                <div style="color:{t['text_muted']}; font-size:0.9rem;">
                    Requiring immediate attention from designated stakeholders
                </div>
            </div>
            """, unsafe_allow_html=True)

            for esc in escalations:
                level_colors = {
                    'L1': t['warning'],
                    'L2': t['danger'],
                    'L3': '#8B0000'  # Dark red
                }
                level_color = level_colors.get(esc['level'], t['warning'])

                st.markdown(f"""
                <div style="background:{t['card']}; border:1px solid {level_color}; border-radius:12px;
                            padding:1rem; margin:0.75rem 0;">
                    <div style="display:flex; justify-content:space-between; align-items:start;">
                        <div style="flex:1;">
                            <div style="display:flex; align-items:center; gap:0.5rem;">
                                <span style="background:{level_color}; color:white; padding:0.2rem 0.5rem;
                                            border-radius:4px; font-weight:700; font-size:0.8rem;">
                                    {esc['level']}
                                </span>
                                <span style="font-weight:700; color:{t['text']};">{esc['kri_name']}</span>
                            </div>
                            <div style="color:{t['text_muted']}; margin:0.5rem 0; font-size:0.9rem;">
                                Breached for {esc['days_breached']} days | Current: {esc['current_value']:.2f} | Threshold: {esc['threshold']:.2f}
                            </div>
                            <div style="color:{t['accent']}; font-size:0.85rem;">
                                Escalated to: <strong>{esc['escalated_to']}</strong>
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:0.75rem; color:{t['text_muted']};">Escalated on</div>
                            <div style="font-weight:600; color:{t['text']};">{esc['escalated_date']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"✅ Acknowledge", key=f"ack_{esc['kri_name']}"):
                        st.success(f"Acknowledged: {esc['kri_name']}")
                with col2:
                    if st.button(f"📧 Notify", key=f"notify_{esc['kri_name']}"):
                        st.info(f"Notification sent to {esc['escalated_to']}")
                with col3:
                    if st.button(f"📝 Add Note", key=f"note_{esc['kri_name']}"):
                        st.text_area("Add investigation note", key=f"note_text_{esc['kri_name']}")
        else:
            st.success("✅ No active escalations. All KRIs within acceptable thresholds.")

        # Escalation Rules
        st.markdown("---")
        st.markdown("### ⚙️ Escalation Rules Configuration")

        st.markdown(f"""
        <div style="background:{t['card']}; border:1px solid {t['border']}; border-radius:12px; padding:1rem;">
            <table style="width:100%; border-collapse:collapse; font-size:0.9rem;">
                <thead>
                    <tr style="background:{t['bg_secondary']};">
                        <th style="padding:0.75rem; text-align:left; color:{t['text_muted']};">Level</th>
                        <th style="padding:0.75rem; text-align:left; color:{t['text_muted']};">Trigger</th>
                        <th style="padding:0.75rem; text-align:left; color:{t['text_muted']};">Escalate To</th>
                        <th style="padding:0.75rem; text-align:left; color:{t['text_muted']};">SLA</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid {t['border']};">
                        <td style="padding:0.75rem;"><span style="background:{t['warning']}; color:white; padding:0.2rem 0.5rem; border-radius:4px; font-weight:600;">L1</span></td>
                        <td style="padding:0.75rem; color:{t['text']};">Threshold breach > 24 hours</td>
                        <td style="padding:0.75rem; color:{t['text']};">Risk Officer</td>
                        <td style="padding:0.75rem; color:{t['text']};">Response within 4 hours</td>
                    </tr>
                    <tr style="border-bottom:1px solid {t['border']};">
                        <td style="padding:0.75rem;"><span style="background:{t['danger']}; color:white; padding:0.2rem 0.5rem; border-radius:4px; font-weight:600;">L2</span></td>
                        <td style="padding:0.75rem; color:{t['text']};">Threshold breach > 72 hours</td>
                        <td style="padding:0.75rem; color:{t['text']};">Head of Risk Management</td>
                        <td style="padding:0.75rem; color:{t['text']};">Response within 2 hours</td>
                    </tr>
                    <tr>
                        <td style="padding:0.75rem;"><span style="background:#8B0000; color:white; padding:0.2rem 0.5rem; border-radius:4px; font-weight:600;">L3</span></td>
                        <td style="padding:0.75rem; color:{t['text']};">Threshold breach > 7 days OR Critical KRI</td>
                        <td style="padding:0.75rem; color:{t['text']};">Chief Risk Officer / Board</td>
                        <td style="padding:0.75rem; color:{t['text']};">Immediate escalation</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    def _calculate_early_warnings(self) -> List[Dict]:
        """Calculate early warning signals based on trend analysis."""
        warnings = []

        for category, indicators in st.session_state.kri_values.items():
            history = st.session_state.kri_history.get(category, {})

            for name, data in indicators.items():
                hist_values = history.get(name, [])
                if len(hist_values) < 3:
                    continue

                value = data['value']
                threshold = data['threshold']
                direction = data['good_direction']
                unit = data['unit']

                # Calculate velocity (rate of change)
                recent_values = hist_values[-6:]
                velocity = (recent_values[-1] - recent_values[0]) / len(recent_values) if len(recent_values) > 1 else 0

                # Predict days to breach
                if threshold > 0 and velocity != 0:
                    if direction == 'lower':
                        if value < threshold and velocity > 0:
                            days_to_breach = int((threshold - value) / velocity)
                            if 0 < days_to_breach <= 90:
                                warnings.append({
                                    'kri_name': name,
                                    'category': category,
                                    'current_value': value,
                                    'threshold': threshold,
                                    'unit': unit,
                                    'velocity': velocity,
                                    'days_to_breach': days_to_breach,
                                    'trend': 'up',
                                    'severity': 'critical' if days_to_breach <= 30 else 'warning',
                                    'message': f"Trending up at {velocity:.3f}/day. Projected to breach in {days_to_breach} days.",
                                    'actions': [
                                        f"Review root cause of increasing {name}",
                                        "Implement immediate mitigation measures",
                                        "Prepare contingency plan for threshold breach",
                                        "Schedule review with risk committee"
                                    ]
                                })
                    elif direction == 'higher':
                        if value > threshold and velocity < 0:
                            days_to_breach = int((value - threshold) / abs(velocity))
                            if 0 < days_to_breach <= 90:
                                warnings.append({
                                    'kri_name': name,
                                    'category': category,
                                    'current_value': value,
                                    'threshold': threshold,
                                    'unit': unit,
                                    'velocity': velocity,
                                    'days_to_breach': days_to_breach,
                                    'trend': 'down',
                                    'severity': 'critical' if days_to_breach <= 30 else 'warning',
                                    'message': f"Trending down at {velocity:.3f}/day. Projected to breach in {days_to_breach} days.",
                                    'actions': [
                                        f"Investigate declining {name}",
                                        "Identify contributing factors",
                                        "Implement corrective actions",
                                        "Monitor daily until stabilized"
                                    ]
                                })

        return sorted(warnings, key=lambda x: x['days_to_breach'])

    def _calculate_threshold_adjustments(self) -> List[Dict]:
        """Calculate trend-adjusted thresholds."""
        adjustments = []

        for category, indicators in st.session_state.kri_values.items():
            history = st.session_state.kri_history.get(category, {})

            for name, data in indicators.items():
                hist_values = history.get(name, [])
                threshold = data['threshold']

                if threshold > 0 and len(hist_values) >= 6:
                    # Calculate volatility
                    volatility = np.std(hist_values) if len(hist_values) > 1 else 0
                    avg_value = np.mean(hist_values)

                    # Adjust threshold based on volatility (tighter for volatile KRIs)
                    vol_factor = volatility / avg_value if avg_value > 0 else 0
                    adjustment = 1 - (vol_factor * 0.5)  # Reduce threshold for volatile KRIs
                    adjusted_threshold = threshold * max(0.8, min(1.2, adjustment))

                    adjustments.append({
                        'kri_name': name,
                        'category': category,
                        'original': threshold,
                        'adjusted': round(adjusted_threshold, 2),
                        'volatility': round(vol_factor * 100, 1)
                    })

        return adjustments

    def _get_current_escalations(self) -> List[Dict]:
        """Get current active escalations."""
        escalations = []

        for category, indicators in st.session_state.kri_values.items():
            for name, data in indicators.items():
                value = data['value']
                threshold = data['threshold']
                direction = data['good_direction']

                is_breached = False
                if threshold > 0:
                    if direction == 'lower' and value > threshold:
                        is_breached = True
                    elif direction == 'higher' and value < threshold:
                        is_breached = True
                elif threshold == 0 and value > 0:
                    is_breached = True

                if is_breached:
                    # Simulate days breached
                    days_breached = random.randint(1, 14)

                    # Determine escalation level
                    if days_breached > 7:
                        level = 'L3'
                        escalated_to = 'Chief Risk Officer'
                    elif days_breached > 3:
                        level = 'L2'
                        escalated_to = 'Head of Risk Management'
                    else:
                        level = 'L1'
                        escalated_to = 'Risk Officer'

                    escalations.append({
                        'kri_name': name,
                        'category': category,
                        'current_value': value,
                        'threshold': threshold,
                        'days_breached': days_breached,
                        'level': level,
                        'escalated_to': escalated_to,
                        'escalated_date': (datetime.now() - timedelta(days=days_breached)).strftime('%Y-%m-%d')
                    })

        return sorted(escalations, key=lambda x: x['days_breached'], reverse=True)


def render():
    """Entry point for the KRI Dashboard page."""
    page = KRIDashboardPage()
    page.render()
