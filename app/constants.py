"""
Application Constants for AURIX.
All hardcoded values should be defined here for easy maintenance.
"""

from typing import Dict, List, Any


# ============================================
# Application Info
# ============================================
APP_NAME = "AURIX"
APP_VERSION = "4.2.0"
APP_TAGLINE = "Intelligent Audit. Elevated Assurance."
APP_DESCRIPTION = """Platform AI komprehensif untuk Internal Audit di industri keuangan Indonesia. 
Menggabungkan metodologi McKinsey dan Big 4 dengan kecerdasan buatan modern. 
Dilengkapi 26+ modul audit profesional termasuk:
🎮 Gamification & Team Achievements
🎛️ Real-time Command Center
🌐 Interactive Risk Universe
🔬 Root Cause Analyzer (Fishbone & 5 Whys)
📌 Kanban Issue Tracker
🧪 AI Lab & Prompt Playground
📆 Visual Audit Timeline
👥 Team Hub & Collaboration"""
APP_AUTHOR = "MS Hadianto"
APP_YEAR = "2025"

# Module count for reference
TOTAL_MODULES = 26


# ============================================
# Theme Colors - Royal Purple & Gold (Premium)
# ============================================
COLORS = {
    "dark": {
        # Backgrounds - Deep Royal Purple
        "bg": "#0d0a1a",
        "bg_secondary": "#1a1428",
        "card": "#231c35",
        "card_hover": "#2d2442",
        "border": "#3d3254",
        # Text - Warm whites
        "text": "#f8f6ff",
        "text_secondary": "#b8a8d4",
        "text_muted": "#7a6a96",
        # Primary - Royal Purple
        "primary": "#8b5cf6",
        "primary_hover": "#7c3aed",
        # Accent - Rich Gold
        "accent": "#f5b642",
        "accent_hover": "#d4a036",
        # Semantic Colors
        "success": "#22c55e",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        # Special - Premium accents
        "gold": "#f5b642",
        "gold_light": "#fcd779",
        "platinum": "#e8e4f0",
        # Sidebar
        "sidebar_bg": "#0f0b1a",
    },
    "light": {
        # Backgrounds - Soft purple-white
        "bg": "#faf8ff",
        "bg_secondary": "#ffffff",
        "card": "#ffffff",
        "card_hover": "#f5f3fa",
        "border": "#e8e0f5",
        # Text
        "text": "#1a1028",
        "text_secondary": "#4a3a6a",
        "text_muted": "#7a6a96",
        # Primary - Royal Purple
        "primary": "#7c3aed",
        "primary_hover": "#6d28d9",
        # Accent - Rich Gold
        "accent": "#d4a036",
        "accent_hover": "#b8922e",
        # Semantic Colors
        "success": "#16a34a",
        "warning": "#d97706",
        "danger": "#dc2626",
        # Special - Premium accents
        "gold": "#d4a036",
        "gold_light": "#e8b84a",
        "platinum": "#6b5b8a",
        # Sidebar
        "sidebar_bg": "#f5f3fa",
    }
}


# ============================================
# Risk Levels
# ============================================
class RiskLevel:
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


RISK_THRESHOLDS = {
    RiskLevel.HIGH: 0.7,
    RiskLevel.MEDIUM: 0.4,
    RiskLevel.LOW: 0.0
}

RISK_COLORS = {
    RiskLevel.HIGH: "danger",
    RiskLevel.MEDIUM: "warning",
    RiskLevel.LOW: "success"
}


# ============================================
# Status Types
# ============================================
class FindingStatus:
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    CLOSED = "Closed"
    OVERDUE = "Overdue"


class ControlEffectiveness:
    EFFECTIVE = "Effective"
    PARTIALLY_EFFECTIVE = "Partially Effective"
    NOT_EFFECTIVE = "Not Effective"


# ============================================
# Document Types
# ============================================
SUPPORTED_DOCUMENT_TYPES = {
    "pdf": "📄 PDF Document",
    "docx": "📝 Word Document",
    "xlsx": "📊 Excel Spreadsheet",
    "csv": "📋 CSV File",
    "txt": "📃 Text File"
}

DOCUMENT_CATEGORIES = [
    "Audit Reports",
    "SOP/Policies",
    "Regulations",
    "Working Papers",
    "Risk Assessment",
    "Financial Data",
    "IT Documentation",
    "Compliance Reports"
]


# ============================================
# LLM Provider Info
# ============================================
LLM_PROVIDER_INFO = {
    "groq": {
        "name": "Groq",
        "description": "🚀 FASTEST FREE API - Llama 3.3, Mixtral",
        "free": True,
        "url": "https://console.groq.com/keys",
        "default_model": "llama-3.3-70b-versatile"
    },
    "together": {
        "name": "Together AI",
        "description": "🆓 Free credits - Llama, Qwen, DeepSeek",
        "free": True,
        "url": "https://api.together.xyz/",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    },
    "google": {
        "name": "Google AI Studio",
        "description": "🆓 Free Gemini 2.0 Flash",
        "free": True,
        "url": "https://aistudio.google.com/app/apikey",
        "default_model": "gemini-2.0-flash-exp"
    },
    "openrouter": {
        "name": "OpenRouter",
        "description": "🌐 Multi-model access - Free & Paid",
        "free": True,
        "url": "https://openrouter.ai/keys",
        "default_model": "google/gemma-2-9b-it:free"
    },
    "ollama": {
        "name": "Ollama (Local)",
        "description": "💻 Run LLMs locally - FREE",
        "free": True,
        "url": "https://ollama.ai/",
        "default_model": "llama3.2"
    },
    "mock": {
        "name": "Mock (Demo)",
        "description": "🧪 Testing without API key",
        "free": True,
        "url": None,
        "default_model": "mock-model"
    }
}


# ============================================
# Audit-Related Constants
# ============================================
AUDIT_PERSONAS = {
    "internal_audit_manager": "Internal Audit Manager",
    "compliance_officer": "Compliance Officer",
    "risk_analyst": "Risk Analyst",
    "it_auditor": "IT Auditor",
    "fraud_examiner": "Fraud Examiner"
}

AUDIT_TEST_TYPES = [
    "Inquiry",
    "Observation",
    "Inspection",
    "Reperformance",
    "Analytical Review",
    "Walkthrough",
    "Substantive Testing",
    "Compliance Testing"
]

FINDING_SEVERITY = {
    "CRITICAL": {"weight": 5, "color": "danger", "icon": "🔴"},
    "HIGH": {"weight": 4, "color": "danger", "icon": "🔴"},
    "MEDIUM": {"weight": 3, "color": "warning", "icon": "🟡"},
    "LOW": {"weight": 2, "color": "success", "icon": "🟢"},
    "INFORMATIONAL": {"weight": 1, "color": "info", "icon": "🔵"}
}


# ============================================
# Working Paper Templates
# ============================================
WORKING_PAPER_TEMPLATES = {
    "Risk Assessment": {
        "sections": [
            "Executive Summary",
            "Inherent Risk Analysis",
            "Control Assessment",
            "Residual Risk",
            "Recommendations"
        ],
        "format": "structured"
    },
    "Testing Workpaper": {
        "sections": [
            "Objective",
            "Scope",
            "Sample Selection",
            "Testing Procedures",
            "Results",
            "Exceptions",
            "Conclusion"
        ],
        "format": "detailed"
    },
    "Interview Notes": {
        "sections": [
            "Interviewee Details",
            "Questions Asked",
            "Responses",
            "Key Observations",
            "Follow-up Actions"
        ],
        "format": "narrative"
    },
    "Process Walkthrough": {
        "sections": [
            "Process Overview",
            "Steps Documented",
            "Key Controls",
            "Control Gaps",
            "Recommendations"
        ],
        "format": "flowchart"
    },
    "Exception Report": {
        "sections": [
            "Exception Summary",
            "Root Cause Analysis",
            "Impact Assessment",
            "Management Response",
            "Remediation Plan"
        ],
        "format": "structured"
    },
    "Control Testing": {
        "sections": [
            "Control Description",
            "Test Approach",
            "Sample Details",
            "Test Results",
            "Conclusion"
        ],
        "format": "matrix"
    }
}


# ============================================
# Date/Time Formats
# ============================================
DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DISPLAY_DATE_FORMAT = "%d %B %Y"
DISPLAY_DATETIME_FORMAT = "%d %B %Y, %H:%M"


# ============================================
# Pagination
# ============================================
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 100


# ============================================
# File Size Limits
# ============================================
MAX_FILE_SIZE_MB = 50
MAX_FILES_PER_UPLOAD = 10


# ============================================
# Session Timeout
# ============================================
SESSION_TIMEOUT_MINUTES = 60
