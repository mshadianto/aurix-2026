-- ============================================
-- AURIX 2026 Database Schema for Supabase
-- Run this in Supabase SQL Editor
-- ============================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 1. USER MANAGEMENT & RBAC
-- ============================================

-- User roles enum
CREATE TYPE user_role AS ENUM (
    'executive',
    'audit_manager',
    'field_auditor',
    'auditee',
    'system_admin'
);

-- Users table (extends Supabase auth.users)
CREATE TABLE public.users (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    role user_role NOT NULL DEFAULT 'field_auditor',
    department TEXT,
    phone TEXT,
    avatar_url TEXT,
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- User sessions for activity tracking
CREATE TABLE public.user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    ip_address INET,
    user_agent TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true
);

-- Audit log for security
CREATE TABLE public.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.users(id),
    action TEXT NOT NULL,
    entity_type TEXT,
    entity_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- 2. AUDIT MANAGEMENT
-- ============================================

-- Audit status enum
CREATE TYPE audit_status AS ENUM (
    'planning',
    'fieldwork',
    'review',
    'reporting',
    'closed'
);

-- Audit engagements
CREATE TABLE public.audit_engagements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    audit_code TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    audit_type TEXT NOT NULL, -- operational, compliance, financial, IT
    risk_level TEXT DEFAULT 'medium', -- low, medium, high, critical
    status audit_status DEFAULT 'planning',
    auditee_department TEXT,
    lead_auditor_id UUID REFERENCES public.users(id),
    planned_start DATE,
    planned_end DATE,
    actual_start DATE,
    actual_end DATE,
    budget_hours NUMERIC(10,2),
    actual_hours NUMERIC(10,2),
    created_by UUID REFERENCES public.users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit team members
CREATE TABLE public.audit_team_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    engagement_id UUID REFERENCES public.audit_engagements(id) ON DELETE CASCADE,
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    role TEXT NOT NULL, -- lead, senior, staff, specialist
    assigned_hours NUMERIC(10,2),
    actual_hours NUMERIC(10,2) DEFAULT 0,
    assigned_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(engagement_id, user_id)
);

-- ============================================
-- 3. WORKPAPERS & DOCUMENTS
-- ============================================

-- Workpaper status enum
CREATE TYPE workpaper_status AS ENUM (
    'draft',
    'pending_review',
    'reviewed',
    'approved',
    'rejected'
);

-- Workpapers
CREATE TABLE public.workpapers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    engagement_id UUID REFERENCES public.audit_engagements(id) ON DELETE CASCADE,
    reference_no TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    objective TEXT,
    procedure TEXT,
    conclusion TEXT,
    status workpaper_status DEFAULT 'draft',
    prepared_by UUID REFERENCES public.users(id),
    reviewed_by UUID REFERENCES public.users(id),
    approved_by UUID REFERENCES public.users(id),
    prepared_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ,
    approved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Documents/Attachments
CREATE TABLE public.documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    engagement_id UUID REFERENCES public.audit_engagements(id) ON DELETE CASCADE,
    workpaper_id UUID REFERENCES public.workpapers(id) ON DELETE SET NULL,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT,
    file_type TEXT,
    description TEXT,
    uploaded_by UUID REFERENCES public.users(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- 4. FINDINGS & ISSUES
-- ============================================

-- Finding severity enum
CREATE TYPE finding_severity AS ENUM (
    'low',
    'medium',
    'high',
    'critical'
);

-- Finding status enum
CREATE TYPE finding_status AS ENUM (
    'draft',
    'pending_review',
    'confirmed',
    'disputed',
    'closed'
);

-- Audit findings
CREATE TABLE public.findings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    engagement_id UUID REFERENCES public.audit_engagements(id) ON DELETE CASCADE,
    workpaper_id UUID REFERENCES public.workpapers(id) ON DELETE SET NULL,
    finding_code TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    condition TEXT NOT NULL,
    criteria TEXT NOT NULL,
    cause TEXT,
    effect TEXT,
    recommendation TEXT,
    management_response TEXT,
    severity finding_severity DEFAULT 'medium',
    status finding_status DEFAULT 'draft',
    risk_category TEXT, -- operational, compliance, financial, IT, fraud
    identified_by UUID REFERENCES public.users(id),
    reviewed_by UUID REFERENCES public.users(id),
    auditee_id UUID REFERENCES public.users(id),
    due_date DATE,
    closed_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Action plans for findings
CREATE TABLE public.action_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    finding_id UUID REFERENCES public.findings(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    responsible_id UUID REFERENCES public.users(id),
    target_date DATE,
    completion_date DATE,
    status TEXT DEFAULT 'pending', -- pending, in_progress, completed, overdue
    evidence_path TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- 5. RISK & KRI MONITORING
-- ============================================

-- KRI categories
CREATE TABLE public.kri_categories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- KRI definitions
CREATE TABLE public.kri_definitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    category_id UUID REFERENCES public.kri_categories(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    unit TEXT DEFAULT '%',
    threshold_green NUMERIC(15,4),
    threshold_yellow NUMERIC(15,4),
    threshold_red NUMERIC(15,4),
    good_direction TEXT DEFAULT 'lower', -- lower, higher, optimal
    frequency TEXT DEFAULT 'monthly', -- daily, weekly, monthly, quarterly
    data_source TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- KRI values (historical data)
CREATE TABLE public.kri_values (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kri_id UUID REFERENCES public.kri_definitions(id) ON DELETE CASCADE,
    value NUMERIC(15,4) NOT NULL,
    period_date DATE NOT NULL,
    status TEXT, -- green, yellow, red
    notes TEXT,
    recorded_by UUID REFERENCES public.users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(kri_id, period_date)
);

-- ============================================
-- 6. STRESS TESTING
-- ============================================

-- Stress test scenarios
CREATE TABLE public.stress_scenarios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    description TEXT,
    severity TEXT NOT NULL, -- mild, moderate, severe, extreme
    bi_rate_shock NUMERIC(10,4),
    usdidr_shock NUMERIC(10,4),
    gdp_shock NUMERIC(10,4),
    npl_multiplier NUMERIC(10,4),
    is_predefined BOOLEAN DEFAULT false,
    created_by UUID REFERENCES public.users(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Stress test results
CREATE TABLE public.stress_test_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scenario_id UUID REFERENCES public.stress_scenarios(id) ON DELETE CASCADE,
    run_date TIMESTAMPTZ DEFAULT NOW(),
    initial_car NUMERIC(10,4),
    projected_car NUMERIC(10,4),
    initial_npl NUMERIC(10,4),
    projected_npl NUMERIC(10,4),
    total_loss NUMERIC(20,2),
    outcome TEXT, -- pass, warning, breach
    narrative TEXT,
    run_by UUID REFERENCES public.users(id),
    parameters JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Monte Carlo simulation results
CREATE TABLE public.monte_carlo_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_date TIMESTAMPTZ DEFAULT NOW(),
    num_simulations INTEGER,
    car_mean NUMERIC(10,4),
    car_std NUMERIC(10,4),
    car_min NUMERIC(10,4),
    car_max NUMERIC(10,4),
    var_95 NUMERIC(20,2),
    var_99 NUMERIC(20,2),
    prob_below_8 NUMERIC(10,6),
    risk_rating TEXT,
    run_by UUID REFERENCES public.users(id),
    parameters JSONB,
    car_distribution NUMERIC[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- 7. CONTINUOUS AUDIT & ALERTS
-- ============================================

-- Continuous audit rules
CREATE TABLE public.ca_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    description TEXT,
    category TEXT NOT NULL, -- AML, Financial, Security, Operations
    rule_type TEXT NOT NULL, -- threshold, pattern, anomaly
    severity TEXT DEFAULT 'medium',
    threshold_value NUMERIC(20,4),
    threshold_operator TEXT, -- gt, lt, eq, ne
    is_enabled BOOLEAN DEFAULT true,
    notification_channels TEXT[], -- email, sms, dashboard
    created_by UUID REFERENCES public.users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Continuous audit alerts
CREATE TABLE public.ca_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_id UUID REFERENCES public.ca_rules(id) ON DELETE CASCADE,
    alert_code TEXT UNIQUE NOT NULL,
    severity TEXT NOT NULL,
    status TEXT DEFAULT 'new', -- new, reviewed, escalated, closed
    description TEXT,
    details JSONB,
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_by UUID REFERENCES public.users(id),
    reviewed_at TIMESTAMPTZ,
    closed_by UUID REFERENCES public.users(id),
    closed_at TIMESTAMPTZ,
    notes TEXT
);

-- ============================================
-- 8. FRAUD DETECTION
-- ============================================

-- Fraud alerts from anti-fraud agent
CREATE TABLE public.fraud_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_code TEXT UNIQUE NOT NULL,
    detection_type TEXT NOT NULL, -- structuring, large_cash, rapid_movement, etc.
    severity TEXT NOT NULL,
    status TEXT DEFAULT 'new',
    customer_id TEXT,
    transaction_ids TEXT[],
    total_amount NUMERIC(20,2),
    risk_score NUMERIC(5,2),
    explanation TEXT,
    sar_generated BOOLEAN DEFAULT false,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_by UUID REFERENCES public.users(id),
    reviewed_at TIMESTAMPTZ,
    details JSONB
);

-- ============================================
-- 9. BENCHMARKING (IJK)
-- ============================================

-- Industry benchmarks
CREATE TABLE public.industry_benchmarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    institution_type TEXT NOT NULL, -- BUKU1, BUKU2, BUKU3, BUKU4
    period TEXT NOT NULL, -- 2024-Q4
    industry_mean NUMERIC(15,4),
    industry_median NUMERIC(15,4),
    industry_p25 NUMERIC(15,4),
    industry_p75 NUMERIC(15,4),
    regulatory_min NUMERIC(15,4),
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(metric_id, institution_type, period)
);

-- Entity benchmark results
CREATE TABLE public.benchmark_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_name TEXT NOT NULL,
    institution_type TEXT NOT NULL,
    period TEXT NOT NULL,
    metric_id TEXT NOT NULL,
    entity_value NUMERIC(15,4),
    percentile_rank NUMERIC(5,2),
    status TEXT, -- excellent, good, average, below_average, concern
    deviation_from_mean NUMERIC(15,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- 10. AI/LLM ANALYTICS
-- ============================================

-- LLM usage logs
CREATE TABLE public.llm_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.users(id),
    provider TEXT NOT NULL, -- openai, anthropic, ollama
    model TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    feature TEXT, -- copilot, rag, analysis
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RAG queries and results
CREATE TABLE public.rag_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.users(id),
    query TEXT NOT NULL,
    context_chunks TEXT[],
    response TEXT,
    faithfulness_score NUMERIC(5,4),
    relevance_score NUMERIC(5,4),
    latency_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================

-- Enable RLS on all tables
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_engagements ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.workpapers ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.findings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

-- Users can view their own profile
CREATE POLICY "Users can view own profile"
    ON public.users FOR SELECT
    USING (auth.uid() = id);

-- Admins can view all users
CREATE POLICY "Admins can view all users"
    ON public.users FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.users
            WHERE id = auth.uid() AND role = 'system_admin'
        )
    );

-- Audit team members can view their engagements
CREATE POLICY "Team members can view engagements"
    ON public.audit_engagements FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.audit_team_members
            WHERE engagement_id = audit_engagements.id
            AND user_id = auth.uid()
        )
        OR
        EXISTS (
            SELECT 1 FROM public.users
            WHERE id = auth.uid() AND role IN ('audit_manager', 'executive', 'system_admin')
        )
    );

-- ============================================
-- INDEXES FOR PERFORMANCE
-- ============================================

CREATE INDEX idx_users_role ON public.users(role);
CREATE INDEX idx_users_email ON public.users(email);
CREATE INDEX idx_audit_engagements_status ON public.audit_engagements(status);
CREATE INDEX idx_audit_engagements_lead ON public.audit_engagements(lead_auditor_id);
CREATE INDEX idx_workpapers_engagement ON public.workpapers(engagement_id);
CREATE INDEX idx_workpapers_status ON public.workpapers(status);
CREATE INDEX idx_findings_engagement ON public.findings(engagement_id);
CREATE INDEX idx_findings_severity ON public.findings(severity);
CREATE INDEX idx_findings_status ON public.findings(status);
CREATE INDEX idx_kri_values_kri ON public.kri_values(kri_id);
CREATE INDEX idx_kri_values_date ON public.kri_values(period_date);
CREATE INDEX idx_ca_alerts_status ON public.ca_alerts(status);
CREATE INDEX idx_ca_alerts_rule ON public.ca_alerts(rule_id);
CREATE INDEX idx_fraud_alerts_status ON public.fraud_alerts(status);
CREATE INDEX idx_llm_logs_user ON public.llm_logs(user_id);
CREATE INDEX idx_llm_logs_created ON public.llm_logs(created_at);
CREATE INDEX idx_audit_logs_user ON public.audit_logs(user_id);
CREATE INDEX idx_audit_logs_created ON public.audit_logs(created_at);

-- ============================================
-- FUNCTIONS & TRIGGERS
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to tables with updated_at
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON public.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_engagements_updated_at
    BEFORE UPDATE ON public.audit_engagements
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_workpapers_updated_at
    BEFORE UPDATE ON public.workpapers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_findings_updated_at
    BEFORE UPDATE ON public.findings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Function to log audit actions
CREATE OR REPLACE FUNCTION log_audit_action()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.audit_logs (
        user_id,
        action,
        entity_type,
        entity_id,
        old_values,
        new_values
    ) VALUES (
        auth.uid(),
        TG_OP,
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        CASE WHEN TG_OP = 'DELETE' THEN row_to_json(OLD) ELSE NULL END,
        CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN row_to_json(NEW) ELSE NULL END
    );
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Apply audit logging to critical tables
CREATE TRIGGER audit_findings_changes
    AFTER INSERT OR UPDATE OR DELETE ON public.findings
    FOR EACH ROW EXECUTE FUNCTION log_audit_action();

CREATE TRIGGER audit_workpapers_changes
    AFTER INSERT OR UPDATE OR DELETE ON public.workpapers
    FOR EACH ROW EXECUTE FUNCTION log_audit_action();

-- ============================================
-- SEED DATA
-- ============================================

-- Insert KRI categories
INSERT INTO public.kri_categories (name, description) VALUES
    ('Credit Risk', 'Indicators related to credit quality and loan portfolio'),
    ('Market Risk', 'Indicators related to market movements and trading'),
    ('Liquidity Risk', 'Indicators related to funding and liquidity position'),
    ('Operational Risk', 'Indicators related to operations and processes'),
    ('Compliance Risk', 'Indicators related to regulatory compliance');

-- Insert predefined stress scenarios
INSERT INTO public.stress_scenarios (name, description, severity, bi_rate_shock, usdidr_shock, gdp_shock, npl_multiplier, is_predefined) VALUES
    ('Baseline', 'Normal economic conditions', 'mild', 0, 0, 0, 1.0, true),
    ('Mild Recession', 'Moderate economic slowdown', 'moderate', 100, 10, -2, 1.5, true),
    ('Severe Recession', 'Significant economic contraction', 'severe', 200, 20, -5, 2.5, true),
    ('Financial Crisis', 'Major financial market disruption', 'extreme', 400, 35, -8, 4.0, true),
    ('COVID-19 Scenario', 'Pandemic-style economic shock', 'severe', 150, 15, -3, 2.0, true),
    ('Currency Crisis', 'Sharp currency depreciation', 'severe', 300, 50, -4, 2.0, true);

-- Insert sample CA rules
INSERT INTO public.ca_rules (name, description, category, rule_type, severity, threshold_value, threshold_operator, is_enabled) VALUES
    ('Large Cash Transaction', 'Detect cash transactions above IDR 500M', 'AML', 'threshold', 'high', 500000000, 'gt', true),
    ('Structuring Detection', 'Detect potential structuring patterns', 'AML', 'pattern', 'high', NULL, NULL, true),
    ('Rapid Fund Movement', 'Detect rapid in-out fund movements', 'AML', 'pattern', 'medium', NULL, NULL, true),
    ('High Risk Jurisdiction', 'Transactions involving high-risk countries', 'AML', 'pattern', 'high', NULL, NULL, true),
    ('Dormant Account Reactivation', 'Detect reactivation of dormant accounts', 'AML', 'pattern', 'medium', NULL, NULL, true);

-- ============================================
-- VIEWS FOR REPORTING
-- ============================================

-- View: Engagement summary
CREATE VIEW public.v_engagement_summary AS
SELECT
    e.id,
    e.audit_code,
    e.title,
    e.status,
    e.risk_level,
    e.planned_start,
    e.planned_end,
    u.full_name as lead_auditor,
    COUNT(DISTINCT tm.user_id) as team_size,
    COUNT(DISTINCT f.id) as finding_count,
    COUNT(DISTINCT CASE WHEN f.severity = 'critical' THEN f.id END) as critical_findings
FROM public.audit_engagements e
LEFT JOIN public.users u ON e.lead_auditor_id = u.id
LEFT JOIN public.audit_team_members tm ON e.id = tm.engagement_id
LEFT JOIN public.findings f ON e.id = f.engagement_id
GROUP BY e.id, u.full_name;

-- View: KRI Dashboard
CREATE VIEW public.v_kri_dashboard AS
SELECT
    d.id as kri_id,
    d.name as kri_name,
    c.name as category,
    d.unit,
    v.value as current_value,
    d.threshold_green,
    d.threshold_yellow,
    d.threshold_red,
    d.good_direction,
    v.status,
    v.period_date
FROM public.kri_definitions d
JOIN public.kri_categories c ON d.category_id = c.id
LEFT JOIN LATERAL (
    SELECT * FROM public.kri_values
    WHERE kri_id = d.id
    ORDER BY period_date DESC
    LIMIT 1
) v ON true
WHERE d.is_active = true;

-- View: Alert summary
CREATE VIEW public.v_alert_summary AS
SELECT
    DATE(triggered_at) as alert_date,
    severity,
    status,
    COUNT(*) as count
FROM public.ca_alerts
WHERE triggered_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(triggered_at), severity, status
ORDER BY alert_date DESC;

COMMENT ON SCHEMA public IS 'AURIX 2026 - Internal Audit Excellence Platform';
