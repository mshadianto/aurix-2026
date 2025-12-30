"""
Role-Based Access Control (RBAC) for AURIX 2026.

Defines user roles and their permitted modules/actions based on:
- Executive/Board: Strategic decision-making & risk profile monitoring
- Audit Manager: Audit quality supervision and team management
- Field Auditor: Technical audit execution and evidence collection
- Auditee (Process Owner): Respond to findings and upload remediation evidence
- System Admin: System configuration, security, and AI performance monitoring
"""

from enum import Enum
from typing import Dict, List, Set, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class UserRole(str, Enum):
    """User role definitions."""
    EXECUTIVE = "executive"
    AUDIT_MANAGER = "audit_manager"
    FIELD_AUDITOR = "field_auditor"
    AUDITEE = "auditee"
    SYSTEM_ADMIN = "system_admin"


class AccessLevel(str, Enum):
    """Access level for modules."""
    FULL = "full"          # Read, Write, Delete, Approve
    WRITE = "write"        # Read, Write
    READ_ONLY = "read"     # Read only
    NONE = "none"          # No access


class ModulePermission(BaseModel):
    """Permission configuration for a module."""
    module_id: str
    access_level: AccessLevel = AccessLevel.NONE
    can_approve: bool = False
    can_export: bool = False
    can_configure: bool = False


class RoleConfig(BaseModel):
    """Configuration for a user role."""
    role: UserRole
    display_name: str
    description: str
    permitted_pages: List[str] = Field(default_factory=list)
    module_permissions: Dict[str, ModulePermission] = Field(default_factory=dict)
    strategic_objective: str = ""


# ============================================
# Role-Page Mapping
# ============================================

ROLE_PAGE_ACCESS: Dict[UserRole, List[str]] = {
    UserRole.EXECUTIVE: [
        # Main
        "📊 Dashboard",
        "🏛️ Executive Dashboard",
        "🎛️ Command Center",
        # Risk Intelligence (High-level)
        "📈 KRI Dashboard",
        "🎰 Stress Tester",
        "📊 IJK Benchmarking",
        # Reports
        "📑 Report Builder",
        # Reference
        "⚙️ Settings",
    ],

    UserRole.AUDIT_MANAGER: [
        # Main
        "📊 Dashboard",
        "🏛️ Executive Dashboard",
        "🎛️ Command Center",
        "📋 Findings Tracker",
        # Audit Tools
        "📁 Documents",
        "📝 Workpapers",
        "⚖️ Risk Assessment",
        "📅 Audit Planning",
        "📆 Audit Timeline",
        "🌐 Risk Universe",
        # Monitoring
        "📈 KRI Dashboard",
        "🔄 Continuous Audit",
        "🔍 Fraud Detection",
        # Intelligence
        "📊 Analytics",
        "📑 Report Builder",
        # Collaboration
        "👥 Team Hub",
        "🎮 Gamification",
        # Reference
        "📜 Regulatory RAG",
        "⚙️ Settings",
    ],

    UserRole.FIELD_AUDITOR: [
        # Main
        "📊 Dashboard",
        "🎛️ Command Center",
        "📋 Findings Tracker",
        # Audit Tools
        "📁 Documents",
        "📝 Workpapers",
        "⚖️ Risk Assessment",
        "🎭 PTCF Builder",
        "📌 Issue Tracker",
        "🔬 Root Cause Analyzer",
        "🧮 Sampling Calculator",
        # Monitoring
        "🔍 Fraud Detection",
        "🔄 Process Mining",
        # Intelligence
        "🤖 AI Chat",
        "📊 Analytics",
        # Collaboration
        "👥 Team Hub",
        "🎮 Gamification",
        # Reference
        "📜 Regulatory RAG",
        "⚙️ Settings",
    ],

    UserRole.AUDITEE: [
        # Limited access
        "📊 Dashboard",
        "📋 Findings Tracker",  # Read-only + Action Plan Upload
        # Reference
        "⚙️ Settings",
    ],

    UserRole.SYSTEM_ADMIN: [
        # Full access to all pages
        "📊 Dashboard",
        "🏛️ Executive Dashboard",
        "🎛️ Command Center",
        "📋 Findings Tracker",
        "📁 Documents",
        "📝 Workpapers",
        "⚖️ Risk Assessment",
        "🎭 PTCF Builder",
        "📅 Audit Planning",
        "📆 Audit Timeline",
        "🌐 Risk Universe",
        "📌 Issue Tracker",
        "🔬 Root Cause Analyzer",
        "🧮 Sampling Calculator",
        "📈 KRI Dashboard",
        "🎰 Stress Tester",
        "📊 IJK Benchmarking",
        "🔄 Continuous Audit",
        "🔍 Fraud Detection",
        "🔄 Process Mining",
        "📜 Regulatory RAG",
        "🤖 AI Chat",
        "🧪 AI Lab",
        "📊 Analytics",
        "📑 Report Builder",
        "👥 Team Hub",
        "🎮 Gamification",
        "🔧 Admin Panel",
        "⚙️ Settings",
    ],
}


# ============================================
# Role Configurations
# ============================================

ROLE_CONFIGS: Dict[UserRole, RoleConfig] = {
    UserRole.EXECUTIVE: RoleConfig(
        role=UserRole.EXECUTIVE,
        display_name="Executive / Board",
        description="C-Suite executives and board members",
        permitted_pages=ROLE_PAGE_ACCESS[UserRole.EXECUTIVE],
        strategic_objective="Pengambilan keputusan strategis & monitoring profil risiko"
    ),

    UserRole.AUDIT_MANAGER: RoleConfig(
        role=UserRole.AUDIT_MANAGER,
        display_name="Audit Manager",
        description="Audit team supervisors and managers",
        permitted_pages=ROLE_PAGE_ACCESS[UserRole.AUDIT_MANAGER],
        strategic_objective="Supervisi kualitas audit dan manajemen tim"
    ),

    UserRole.FIELD_AUDITOR: RoleConfig(
        role=UserRole.FIELD_AUDITOR,
        display_name="Field Auditor",
        description="Audit team members performing fieldwork",
        permitted_pages=ROLE_PAGE_ACCESS[UserRole.FIELD_AUDITOR],
        strategic_objective="Eksekusi teknis audit dan pengumpulan bukti"
    ),

    UserRole.AUDITEE: RoleConfig(
        role=UserRole.AUDITEE,
        display_name="Auditee (Process Owner)",
        description="Business unit owners responding to audit findings",
        permitted_pages=ROLE_PAGE_ACCESS[UserRole.AUDITEE],
        strategic_objective="Memberikan respon terhadap temuan dan bukti perbaikan"
    ),

    UserRole.SYSTEM_ADMIN: RoleConfig(
        role=UserRole.SYSTEM_ADMIN,
        display_name="System Admin",
        description="IT administrators managing the AURIX platform",
        permitted_pages=ROLE_PAGE_ACCESS[UserRole.SYSTEM_ADMIN],
        strategic_objective="Pengaturan sistem, keamanan, dan monitoring performa AI"
    ),
}


# ============================================
# Module-Level Permissions
# ============================================

MODULE_PERMISSIONS: Dict[UserRole, Dict[str, AccessLevel]] = {
    UserRole.EXECUTIVE: {
        "findings": AccessLevel.READ_ONLY,
        "workpapers": AccessLevel.NONE,
        "documents": AccessLevel.READ_ONLY,
        "kri": AccessLevel.READ_ONLY,
        "stress_test": AccessLevel.FULL,
        "benchmarking": AccessLevel.FULL,
        "reports": AccessLevel.FULL,
    },

    UserRole.AUDIT_MANAGER: {
        "findings": AccessLevel.FULL,  # Can approve
        "workpapers": AccessLevel.FULL,  # Can review & approve
        "documents": AccessLevel.FULL,
        "kri": AccessLevel.FULL,
        "stress_test": AccessLevel.WRITE,
        "benchmarking": AccessLevel.WRITE,
        "reports": AccessLevel.FULL,
        "planning": AccessLevel.FULL,
        "team": AccessLevel.FULL,
    },

    UserRole.FIELD_AUDITOR: {
        "findings": AccessLevel.WRITE,  # Can create, not approve
        "workpapers": AccessLevel.WRITE,
        "documents": AccessLevel.WRITE,
        "kri": AccessLevel.READ_ONLY,
        "stress_test": AccessLevel.READ_ONLY,
        "benchmarking": AccessLevel.READ_ONLY,
        "reports": AccessLevel.WRITE,
        "planning": AccessLevel.READ_ONLY,
        "fraud_detection": AccessLevel.WRITE,
        "sampling": AccessLevel.FULL,
    },

    UserRole.AUDITEE: {
        "findings": AccessLevel.READ_ONLY,  # Can only view & respond
        "action_plan": AccessLevel.WRITE,  # Can upload action plans
        "documents": AccessLevel.NONE,
        "workpapers": AccessLevel.NONE,
    },

    UserRole.SYSTEM_ADMIN: {
        "findings": AccessLevel.FULL,
        "workpapers": AccessLevel.FULL,
        "documents": AccessLevel.FULL,
        "kri": AccessLevel.FULL,
        "stress_test": AccessLevel.FULL,
        "benchmarking": AccessLevel.FULL,
        "reports": AccessLevel.FULL,
        "planning": AccessLevel.FULL,
        "team": AccessLevel.FULL,
        "admin": AccessLevel.FULL,
        "llm_config": AccessLevel.FULL,
        "database": AccessLevel.FULL,
    },
}


# ============================================
# Helper Functions
# ============================================

def get_user_role() -> UserRole:
    """Get current user's role from session state."""
    import streamlit as st
    return st.session_state.get('user_role', UserRole.FIELD_AUDITOR)


def set_user_role(role: UserRole):
    """Set current user's role in session state."""
    import streamlit as st
    st.session_state.user_role = role


def get_permitted_pages(role: Optional[UserRole] = None) -> List[str]:
    """Get list of pages permitted for a role."""
    if role is None:
        role = get_user_role()
    return ROLE_PAGE_ACCESS.get(role, [])


def can_access_page(page_name: str, role: Optional[UserRole] = None) -> bool:
    """Check if role can access a specific page."""
    if role is None:
        role = get_user_role()
    permitted = get_permitted_pages(role)
    return page_name in permitted


def get_module_access(module_id: str, role: Optional[UserRole] = None) -> AccessLevel:
    """Get access level for a module."""
    if role is None:
        role = get_user_role()
    permissions = MODULE_PERMISSIONS.get(role, {})
    return permissions.get(module_id, AccessLevel.NONE)


def can_write(module_id: str, role: Optional[UserRole] = None) -> bool:
    """Check if role has write access to a module."""
    access = get_module_access(module_id, role)
    return access in [AccessLevel.WRITE, AccessLevel.FULL]


def can_approve(module_id: str, role: Optional[UserRole] = None) -> bool:
    """Check if role can approve items in a module."""
    access = get_module_access(module_id, role)
    return access == AccessLevel.FULL


def get_role_config(role: Optional[UserRole] = None) -> RoleConfig:
    """Get configuration for a role."""
    if role is None:
        role = get_user_role()
    return ROLE_CONFIGS.get(role, ROLE_CONFIGS[UserRole.FIELD_AUDITOR])


def filter_pages_by_role(all_pages: List[str], role: Optional[UserRole] = None) -> List[str]:
    """Filter a list of pages based on role permissions."""
    if role is None:
        role = get_user_role()
    permitted = set(get_permitted_pages(role))
    return [p for p in all_pages if p in permitted]


def get_role_display_info(role: UserRole) -> Dict:
    """Get display information for a role."""
    config = ROLE_CONFIGS.get(role)
    if not config:
        return {}

    return {
        "role": role.value,
        "display_name": config.display_name,
        "description": config.description,
        "objective": config.strategic_objective,
        "page_count": len(config.permitted_pages),
    }


def get_all_roles_info() -> List[Dict]:
    """Get display info for all roles."""
    return [get_role_display_info(role) for role in UserRole]


# ============================================
# Session Management
# ============================================

class UserSession(BaseModel):
    """User session information."""
    user_id: str = Field(default="demo_user")
    username: str = Field(default="Demo User")
    role: UserRole = Field(default=UserRole.FIELD_AUDITOR)
    email: str = Field(default="user@example.com")
    department: str = Field(default="Internal Audit")
    login_time: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)


def get_current_session() -> UserSession:
    """Get current user session."""
    import streamlit as st

    if 'user_session' not in st.session_state:
        st.session_state.user_session = UserSession()

    return st.session_state.user_session


def update_session_role(role: UserRole):
    """Update the role in current session."""
    import streamlit as st

    session = get_current_session()
    session.role = role
    session.last_activity = datetime.now()
    st.session_state.user_session = session
    st.session_state.user_role = role
