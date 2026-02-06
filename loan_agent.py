#!/usr/bin/env python3
"""
Loan Application Agent - AI-powered loan processing and decisioning.

This agent provides comprehensive loan application processing capabilities:
- Application submission and validation
- Credit score evaluation
- Risk assessment and scoring
- Automated approval/rejection decisions
- Interest rate calculations
- Payment schedule generation
- Document verification
- Loan status tracking

USAGE:
    python loan_agent.py                    # Interactive mode
    python loan_agent.py --auto "message"   # Single message mode
"""
import os
import sys
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, TypedDict
from dataclasses import dataclass, asdict, field
from enum import Enum
import random

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=False, write_through=True)

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


# =============================================================================
# Data Models
# =============================================================================

class LoanType(str, Enum):
    PERSONAL = "personal"
    MORTGAGE = "mortgage"
    AUTO = "auto"
    BUSINESS = "business"
    STUDENT = "student"
    HOME_EQUITY = "home_equity"


class LoanStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    PENDING_DOCUMENTS = "pending_documents"
    APPROVED = "approved"
    REJECTED = "rejected"
    FUNDED = "funded"
    CLOSED = "closed"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class Applicant:
    """Loan applicant information."""
    applicant_id: str
    first_name: str
    last_name: str
    email: str
    phone: str
    ssn_hash: str  # Hashed for security
    date_of_birth: str
    address: str
    city: str
    state: str
    zip_code: str
    employment_status: str
    employer_name: str
    job_title: str
    annual_income: float
    monthly_expenses: float
    years_employed: int
    credit_score: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LoanApplication:
    """Loan application details."""
    application_id: str
    applicant_id: str
    loan_type: str
    requested_amount: float
    loan_term_months: int
    purpose: str
    status: str = LoanStatus.DRAFT.value
    interest_rate: Optional[float] = None
    monthly_payment: Optional[float] = None
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    decision_reason: Optional[str] = None
    collateral_type: Optional[str] = None
    collateral_value: Optional[float] = None
    co_applicant_id: Optional[str] = None
    documents: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    approved_at: Optional[str] = None
    funded_at: Optional[str] = None


@dataclass
class CreditReport:
    """Credit report summary."""
    applicant_id: str
    credit_score: int
    credit_bureau: str
    report_date: str
    payment_history: str  # excellent, good, fair, poor
    credit_utilization: float  # percentage
    length_of_history_years: int
    new_accounts_last_year: int
    hard_inquiries_last_year: int
    derogatory_marks: int
    total_accounts: int
    open_accounts: int
    total_balance: float
    total_credit_limit: float


@dataclass
class PaymentSchedule:
    """Loan payment schedule entry."""
    payment_number: int
    due_date: str
    principal: float
    interest: float
    total_payment: float
    remaining_balance: float


# =============================================================================
# In-Memory Data Store
# =============================================================================

class LoanDataStore:
    """In-memory storage for loan data (would be database in production)."""
    
    def __init__(self):
        self.applicants: Dict[str, Applicant] = {}
        self.applications: Dict[str, LoanApplication] = {}
        self.credit_reports: Dict[str, CreditReport] = {}
        self.documents: Dict[str, Dict] = {}
        
        # Initialize with sample data
        self._init_sample_data()
    
    def _init_sample_data(self):
        """Create sample applicants and applications for demo."""
        # Sample applicant 1
        applicant1 = Applicant(
            applicant_id="APP-001",
            first_name="John",
            last_name="Smith",
            email="john.smith@email.com",
            phone="555-123-4567",
            ssn_hash=hashlib.sha256("123-45-6789".encode()).hexdigest()[:16],
            date_of_birth="1985-03-15",
            address="123 Main Street",
            city="Springfield",
            state="IL",
            zip_code="62701",
            employment_status="employed",
            employer_name="Tech Corp Inc",
            job_title="Software Engineer",
            annual_income=95000.00,
            monthly_expenses=3200.00,
            years_employed=5,
            credit_score=742
        )
        self.applicants[applicant1.applicant_id] = applicant1
        
        # Sample credit report for applicant 1
        credit1 = CreditReport(
            applicant_id="APP-001",
            credit_score=742,
            credit_bureau="Experian",
            report_date=datetime.now().isoformat(),
            payment_history="excellent",
            credit_utilization=0.23,
            length_of_history_years=12,
            new_accounts_last_year=1,
            hard_inquiries_last_year=2,
            derogatory_marks=0,
            total_accounts=8,
            open_accounts=5,
            total_balance=15000.00,
            total_credit_limit=65000.00
        )
        self.credit_reports["APP-001"] = credit1
        
        # Sample loan application
        app1 = LoanApplication(
            application_id="LA-2024-001",
            applicant_id="APP-001",
            loan_type=LoanType.PERSONAL.value,
            requested_amount=25000.00,
            loan_term_months=60,
            purpose="Home improvement",
            status=LoanStatus.SUBMITTED.value
        )
        self.applications[app1.application_id] = app1
        
        # Sample applicant 2 (higher risk)
        applicant2 = Applicant(
            applicant_id="APP-002",
            first_name="Jane",
            last_name="Doe",
            email="jane.doe@email.com",
            phone="555-987-6543",
            ssn_hash=hashlib.sha256("987-65-4321".encode()).hexdigest()[:16],
            date_of_birth="1992-07-22",
            address="456 Oak Avenue",
            city="Chicago",
            state="IL",
            zip_code="60601",
            employment_status="self-employed",
            employer_name="Freelance Consulting",
            job_title="Consultant",
            annual_income=65000.00,
            monthly_expenses=2800.00,
            years_employed=2,
            credit_score=628
        )
        self.applicants[applicant2.applicant_id] = applicant2
        
        credit2 = CreditReport(
            applicant_id="APP-002",
            credit_score=628,
            credit_bureau="TransUnion",
            report_date=datetime.now().isoformat(),
            payment_history="fair",
            credit_utilization=0.67,
            length_of_history_years=5,
            new_accounts_last_year=3,
            hard_inquiries_last_year=5,
            derogatory_marks=1,
            total_accounts=6,
            open_accounts=4,
            total_balance=28000.00,
            total_credit_limit=42000.00
        )
        self.credit_reports["APP-002"] = credit2


# Global data store
_data_store: Optional[LoanDataStore] = None


def get_data_store() -> LoanDataStore:
    """Get or create the data store."""
    global _data_store
    if _data_store is None:
        _data_store = LoanDataStore()
    return _data_store


# =============================================================================
# Loan Calculation Functions
# =============================================================================

def calculate_monthly_payment(principal: float, annual_rate: float, term_months: int) -> float:
    """Calculate monthly payment using amortization formula."""
    if annual_rate == 0:
        return principal / term_months
    
    monthly_rate = annual_rate / 12 / 100
    payment = principal * (monthly_rate * (1 + monthly_rate)**term_months) / \
              ((1 + monthly_rate)**term_months - 1)
    return round(payment, 2)


def calculate_total_interest(principal: float, monthly_payment: float, term_months: int) -> float:
    """Calculate total interest over loan term."""
    total_paid = monthly_payment * term_months
    return round(total_paid - principal, 2)


def generate_amortization_schedule(
    principal: float,
    annual_rate: float,
    term_months: int,
    start_date: datetime = None
) -> List[PaymentSchedule]:
    """Generate full amortization schedule."""
    if start_date is None:
        start_date = datetime.now() + timedelta(days=30)
    
    monthly_rate = annual_rate / 12 / 100
    monthly_payment = calculate_monthly_payment(principal, annual_rate, term_months)
    
    schedule = []
    balance = principal
    
    for i in range(1, term_months + 1):
        interest = round(balance * monthly_rate, 2)
        principal_paid = round(monthly_payment - interest, 2)
        balance = round(balance - principal_paid, 2)
        
        if balance < 0:
            balance = 0
        
        payment_date = start_date + timedelta(days=30 * i)
        
        schedule.append(PaymentSchedule(
            payment_number=i,
            due_date=payment_date.strftime("%Y-%m-%d"),
            principal=principal_paid,
            interest=interest,
            total_payment=monthly_payment,
            remaining_balance=balance
        ))
    
    return schedule


def get_base_interest_rate(loan_type: str, credit_score: int) -> float:
    """Get base interest rate based on loan type and credit score."""
    # Base rates by loan type
    base_rates = {
        LoanType.PERSONAL.value: 10.0,
        LoanType.MORTGAGE.value: 6.5,
        LoanType.AUTO.value: 7.0,
        LoanType.BUSINESS.value: 8.5,
        LoanType.STUDENT.value: 5.5,
        LoanType.HOME_EQUITY.value: 7.5,
    }
    
    base = base_rates.get(loan_type, 10.0)
    
    # Adjust based on credit score
    if credit_score >= 800:
        adjustment = -2.0
    elif credit_score >= 750:
        adjustment = -1.0
    elif credit_score >= 700:
        adjustment = 0.0
    elif credit_score >= 650:
        adjustment = 2.0
    elif credit_score >= 600:
        adjustment = 4.0
    else:
        adjustment = 7.0
    
    return round(base + adjustment, 2)


def calculate_dti_ratio(monthly_income: float, monthly_debt: float, new_payment: float) -> float:
    """Calculate debt-to-income ratio."""
    total_debt = monthly_debt + new_payment
    return round((total_debt / monthly_income) * 100, 2)


def assess_risk(
    credit_score: int,
    dti_ratio: float,
    employment_years: int,
    loan_to_income_ratio: float
) -> tuple[float, RiskLevel]:
    """Assess loan risk and return score (0-100) and level."""
    score = 100.0
    
    # Credit score impact (40%)
    if credit_score >= 800:
        score -= 0
    elif credit_score >= 750:
        score -= 5
    elif credit_score >= 700:
        score -= 15
    elif credit_score >= 650:
        score -= 30
    elif credit_score >= 600:
        score -= 45
    else:
        score -= 60
    
    # DTI impact (30%)
    if dti_ratio <= 20:
        score -= 0
    elif dti_ratio <= 30:
        score -= 5
    elif dti_ratio <= 40:
        score -= 15
    elif dti_ratio <= 50:
        score -= 25
    else:
        score -= 40
    
    # Employment stability (15%)
    if employment_years >= 5:
        score -= 0
    elif employment_years >= 3:
        score -= 5
    elif employment_years >= 1:
        score -= 10
    else:
        score -= 20
    
    # Loan-to-income ratio (15%)
    if loan_to_income_ratio <= 1:
        score -= 0
    elif loan_to_income_ratio <= 2:
        score -= 5
    elif loan_to_income_ratio <= 3:
        score -= 10
    else:
        score -= 20
    
    score = max(0, min(100, score))
    
    # Determine risk level
    if score >= 80:
        level = RiskLevel.LOW
    elif score >= 60:
        level = RiskLevel.MODERATE
    elif score >= 40:
        level = RiskLevel.HIGH
    else:
        level = RiskLevel.VERY_HIGH
    
    return round(score, 1), level


# =============================================================================
# Loan Tools
# =============================================================================

@tool
def create_applicant(
    first_name: str,
    last_name: str,
    email: str,
    phone: str,
    date_of_birth: str,
    ssn_last_four: str,
    address: str,
    city: str,
    state: str,
    zip_code: str,
    employment_status: str,
    employer_name: str,
    job_title: str,
    annual_income: float,
    monthly_expenses: float,
    years_employed: int
) -> str:
    """Create a new loan applicant profile.
    
    Args:
        first_name: Applicant's first name
        last_name: Applicant's last name
        email: Email address
        phone: Phone number
        date_of_birth: Date of birth (YYYY-MM-DD)
        ssn_last_four: Last 4 digits of SSN (for verification)
        address: Street address
        city: City
        state: State (2-letter code)
        zip_code: ZIP code
        employment_status: employed, self-employed, unemployed, retired
        employer_name: Current employer name
        job_title: Current job title
        annual_income: Annual gross income
        monthly_expenses: Monthly expenses (rent, utilities, etc.)
        years_employed: Years at current employer
        
    Returns:
        Applicant ID and profile summary
    """
    store = get_data_store()
    
    # Generate applicant ID
    applicant_id = f"APP-{str(uuid.uuid4())[:8].upper()}"
    
    # Hash SSN for security
    ssn_hash = hashlib.sha256(ssn_last_four.encode()).hexdigest()[:16]
    
    applicant = Applicant(
        applicant_id=applicant_id,
        first_name=first_name,
        last_name=last_name,
        email=email,
        phone=phone,
        ssn_hash=ssn_hash,
        date_of_birth=date_of_birth,
        address=address,
        city=city,
        state=state,
        zip_code=zip_code,
        employment_status=employment_status,
        employer_name=employer_name,
        job_title=job_title,
        annual_income=annual_income,
        monthly_expenses=monthly_expenses,
        years_employed=years_employed
    )
    
    store.applicants[applicant_id] = applicant
    
    output = f"## ✅ Applicant Created\n\n"
    output += f"**Applicant ID**: `{applicant_id}`\n\n"
    output += f"**Name**: {first_name} {last_name}\n"
    output += f"**Email**: {email}\n"
    output += f"**Employment**: {job_title} at {employer_name}\n"
    output += f"**Annual Income**: ${annual_income:,.2f}\n"
    output += f"**Monthly Expenses**: ${monthly_expenses:,.2f}\n\n"
    output += "**Next Steps**:\n"
    output += f"1. Run credit check: `check_credit_score(applicant_id='{applicant_id}')`\n"
    output += f"2. Submit application: `submit_loan_application(applicant_id='{applicant_id}', ...)`"
    
    return output


@tool
def get_applicant_info(applicant_id: str) -> str:
    """Get detailed information about an applicant.
    
    Args:
        applicant_id: The applicant's ID (e.g., APP-001)
        
    Returns:
        Applicant details
    """
    store = get_data_store()
    
    if applicant_id not in store.applicants:
        return f"ERROR: Applicant '{applicant_id}' not found. Use list_applicants() to see available applicants."
    
    applicant = store.applicants[applicant_id]
    credit = store.credit_reports.get(applicant_id)
    
    output = f"## Applicant Profile: {applicant_id}\n\n"
    output += f"### Personal Information\n"
    output += f"- **Name**: {applicant.first_name} {applicant.last_name}\n"
    output += f"- **Email**: {applicant.email}\n"
    output += f"- **Phone**: {applicant.phone}\n"
    output += f"- **DOB**: {applicant.date_of_birth}\n"
    output += f"- **Address**: {applicant.address}, {applicant.city}, {applicant.state} {applicant.zip_code}\n\n"
    
    output += f"### Employment\n"
    output += f"- **Status**: {applicant.employment_status}\n"
    output += f"- **Employer**: {applicant.employer_name}\n"
    output += f"- **Title**: {applicant.job_title}\n"
    output += f"- **Years**: {applicant.years_employed}\n\n"
    
    output += f"### Financial\n"
    output += f"- **Annual Income**: ${applicant.annual_income:,.2f}\n"
    output += f"- **Monthly Income**: ${applicant.annual_income/12:,.2f}\n"
    output += f"- **Monthly Expenses**: ${applicant.monthly_expenses:,.2f}\n"
    output += f"- **Disposable Income**: ${applicant.annual_income/12 - applicant.monthly_expenses:,.2f}/month\n\n"
    
    if credit:
        output += f"### Credit Summary\n"
        output += f"- **Score**: {credit.credit_score}\n"
        output += f"- **Bureau**: {credit.credit_bureau}\n"
        output += f"- **Utilization**: {credit.credit_utilization*100:.1f}%\n"
        output += f"- **Payment History**: {credit.payment_history}\n"
    else:
        output += "### Credit\n"
        output += "- No credit report on file. Run `check_credit_score` first.\n"
    
    return output


@tool
def list_applicants() -> str:
    """List all applicants in the system.
    
    Returns:
        List of applicants with basic info
    """
    store = get_data_store()
    
    if not store.applicants:
        return "No applicants found. Create one with `create_applicant()`."
    
    output = "## Applicants\n\n"
    output += "| ID | Name | Income | Credit Score | Status |\n"
    output += "|---|---|---|---|---|\n"
    
    for app_id, applicant in store.applicants.items():
        credit = store.credit_reports.get(app_id)
        score = credit.credit_score if credit else "N/A"
        output += f"| {app_id} | {applicant.first_name} {applicant.last_name} | "
        output += f"${applicant.annual_income:,.0f} | {score} | {applicant.employment_status} |\n"
    
    return output


@tool
def check_credit_score(applicant_id: str) -> str:
    """Check/pull credit score for an applicant.
    
    This simulates pulling a credit report from a bureau.
    
    Args:
        applicant_id: The applicant's ID
        
    Returns:
        Credit score and report summary
    """
    store = get_data_store()
    
    if applicant_id not in store.applicants:
        return f"ERROR: Applicant '{applicant_id}' not found."
    
    applicant = store.applicants[applicant_id]
    
    # If already have a report, return it
    if applicant_id in store.credit_reports:
        credit = store.credit_reports[applicant_id]
    else:
        # Simulate credit pull (generate realistic data)
        base_score = random.randint(580, 820)
        
        credit = CreditReport(
            applicant_id=applicant_id,
            credit_score=base_score,
            credit_bureau=random.choice(["Experian", "TransUnion", "Equifax"]),
            report_date=datetime.now().isoformat(),
            payment_history=("excellent" if base_score >= 750 else 
                           "good" if base_score >= 700 else 
                           "fair" if base_score >= 650 else "poor"),
            credit_utilization=round(random.uniform(0.1, 0.8), 2),
            length_of_history_years=random.randint(2, 20),
            new_accounts_last_year=random.randint(0, 4),
            hard_inquiries_last_year=random.randint(0, 6),
            derogatory_marks=0 if base_score >= 700 else random.randint(0, 3),
            total_accounts=random.randint(3, 15),
            open_accounts=random.randint(2, 8),
            total_balance=round(random.uniform(5000, 50000), 2),
            total_credit_limit=round(random.uniform(20000, 100000), 2)
        )
        
        store.credit_reports[applicant_id] = credit
        applicant.credit_score = base_score
    
    output = f"## 📊 Credit Report for {applicant.first_name} {applicant.last_name}\n\n"
    output += f"**Credit Score**: {credit.credit_score}"
    
    if credit.credit_score >= 750:
        output += " 🟢 Excellent\n"
    elif credit.credit_score >= 700:
        output += " 🟡 Good\n"
    elif credit.credit_score >= 650:
        output += " 🟠 Fair\n"
    else:
        output += " 🔴 Poor\n"
    
    output += f"**Bureau**: {credit.credit_bureau}\n"
    output += f"**Report Date**: {credit.report_date[:10]}\n\n"
    
    output += "### Account Summary\n"
    output += f"- Total Accounts: {credit.total_accounts}\n"
    output += f"- Open Accounts: {credit.open_accounts}\n"
    output += f"- Total Balance: ${credit.total_balance:,.2f}\n"
    output += f"- Credit Limit: ${credit.total_credit_limit:,.2f}\n"
    output += f"- Utilization: {credit.credit_utilization*100:.1f}%\n\n"
    
    output += "### Risk Factors\n"
    output += f"- Payment History: {credit.payment_history}\n"
    output += f"- History Length: {credit.length_of_history_years} years\n"
    output += f"- New Accounts (12 mo): {credit.new_accounts_last_year}\n"
    output += f"- Hard Inquiries (12 mo): {credit.hard_inquiries_last_year}\n"
    output += f"- Derogatory Marks: {credit.derogatory_marks}\n"
    
    return output


@tool
def submit_loan_application(
    applicant_id: str,
    loan_type: str,
    requested_amount: float,
    loan_term_months: int,
    purpose: str,
    collateral_type: str = None,
    collateral_value: float = None
) -> str:
    """Submit a new loan application.
    
    Args:
        applicant_id: The applicant's ID
        loan_type: Type of loan (personal, mortgage, auto, business, student, home_equity)
        requested_amount: Loan amount requested
        loan_term_months: Loan term in months (12, 24, 36, 48, 60, 120, 180, 240, 360)
        purpose: Purpose of the loan
        collateral_type: Type of collateral (for secured loans)
        collateral_value: Value of collateral
        
    Returns:
        Application ID and summary
    """
    store = get_data_store()
    
    if applicant_id not in store.applicants:
        return f"ERROR: Applicant '{applicant_id}' not found."
    
    # Validate loan type
    valid_types = [lt.value for lt in LoanType]
    if loan_type.lower() not in valid_types:
        return f"ERROR: Invalid loan type. Valid types: {', '.join(valid_types)}"
    
    # Generate application ID
    app_id = f"LA-{datetime.now().year}-{str(uuid.uuid4())[:6].upper()}"
    
    application = LoanApplication(
        application_id=app_id,
        applicant_id=applicant_id,
        loan_type=loan_type.lower(),
        requested_amount=requested_amount,
        loan_term_months=loan_term_months,
        purpose=purpose,
        status=LoanStatus.SUBMITTED.value,
        collateral_type=collateral_type,
        collateral_value=collateral_value
    )
    
    store.applications[app_id] = application
    
    applicant = store.applicants[applicant_id]
    
    output = f"## ✅ Loan Application Submitted\n\n"
    output += f"**Application ID**: `{app_id}`\n\n"
    output += f"### Application Details\n"
    output += f"- **Applicant**: {applicant.first_name} {applicant.last_name}\n"
    output += f"- **Loan Type**: {loan_type.title()}\n"
    output += f"- **Amount**: ${requested_amount:,.2f}\n"
    output += f"- **Term**: {loan_term_months} months ({loan_term_months//12} years)\n"
    output += f"- **Purpose**: {purpose}\n"
    
    if collateral_type:
        output += f"- **Collateral**: {collateral_type} (${collateral_value:,.2f})\n"
    
    output += f"\n**Status**: {LoanStatus.SUBMITTED.value.replace('_', ' ').title()}\n\n"
    output += "**Next Steps**:\n"
    output += f"1. Assess risk: `assess_loan_risk(application_id='{app_id}')`\n"
    output += f"2. Calculate rates: `calculate_interest_rate(application_id='{app_id}')`\n"
    output += f"3. Make decision: `process_loan_decision(application_id='{app_id}')`"
    
    return output


@tool
def get_application_status(application_id: str) -> str:
    """Get the current status of a loan application.
    
    Args:
        application_id: The loan application ID
        
    Returns:
        Application status and details
    """
    store = get_data_store()
    
    if application_id not in store.applications:
        return f"ERROR: Application '{application_id}' not found."
    
    app = store.applications[application_id]
    applicant = store.applicants.get(app.applicant_id)
    
    output = f"## Loan Application: {application_id}\n\n"
    
    # Status badge
    status_emoji = {
        LoanStatus.DRAFT.value: "📝",
        LoanStatus.SUBMITTED.value: "📨",
        LoanStatus.UNDER_REVIEW.value: "🔍",
        LoanStatus.PENDING_DOCUMENTS.value: "📎",
        LoanStatus.APPROVED.value: "✅",
        LoanStatus.REJECTED.value: "❌",
        LoanStatus.FUNDED.value: "💰",
        LoanStatus.CLOSED.value: "🔒"
    }
    
    emoji = status_emoji.get(app.status, "❓")
    output += f"**Status**: {emoji} {app.status.replace('_', ' ').title()}\n\n"
    
    if applicant:
        output += f"**Applicant**: {applicant.first_name} {applicant.last_name} ({app.applicant_id})\n"
    
    output += f"**Loan Type**: {app.loan_type.title()}\n"
    output += f"**Amount**: ${app.requested_amount:,.2f}\n"
    output += f"**Term**: {app.loan_term_months} months\n"
    output += f"**Purpose**: {app.purpose}\n\n"
    
    if app.interest_rate:
        output += f"### Loan Terms\n"
        output += f"- **Interest Rate**: {app.interest_rate}% APR\n"
        output += f"- **Monthly Payment**: ${app.monthly_payment:,.2f}\n"
        total_interest = calculate_total_interest(
            app.requested_amount, app.monthly_payment, app.loan_term_months
        )
        output += f"- **Total Interest**: ${total_interest:,.2f}\n\n"
    
    if app.risk_score is not None:
        output += f"### Risk Assessment\n"
        output += f"- **Risk Score**: {app.risk_score}/100\n"
        output += f"- **Risk Level**: {app.risk_level.title()}\n\n"
    
    if app.decision_reason:
        output += f"### Decision\n"
        output += f"{app.decision_reason}\n\n"
    
    output += f"### Timeline\n"
    output += f"- Created: {app.created_at[:10]}\n"
    output += f"- Updated: {app.updated_at[:10]}\n"
    if app.approved_at:
        output += f"- Approved: {app.approved_at[:10]}\n"
    if app.funded_at:
        output += f"- Funded: {app.funded_at[:10]}\n"
    
    return output


@tool
def list_applications(status: str = None, applicant_id: str = None) -> str:
    """List loan applications with optional filters.
    
    Args:
        status: Filter by status (optional)
        applicant_id: Filter by applicant (optional)
        
    Returns:
        List of applications
    """
    store = get_data_store()
    
    apps = list(store.applications.values())
    
    # Apply filters
    if status:
        apps = [a for a in apps if a.status == status.lower()]
    if applicant_id:
        apps = [a for a in apps if a.applicant_id == applicant_id]
    
    if not apps:
        return "No applications found matching the criteria."
    
    output = "## Loan Applications\n\n"
    output += "| Application ID | Applicant | Type | Amount | Status |\n"
    output += "|---|---|---|---|---|\n"
    
    for app in apps:
        applicant = store.applicants.get(app.applicant_id)
        name = f"{applicant.first_name} {applicant.last_name}" if applicant else app.applicant_id
        output += f"| {app.application_id} | {name} | {app.loan_type} | "
        output += f"${app.requested_amount:,.0f} | {app.status} |\n"
    
    return output


@tool
def assess_loan_risk(application_id: str) -> str:
    """Perform risk assessment on a loan application.
    
    Evaluates credit score, DTI ratio, employment stability, and loan-to-income ratio.
    
    Args:
        application_id: The loan application ID
        
    Returns:
        Risk assessment report
    """
    store = get_data_store()
    
    if application_id not in store.applications:
        return f"ERROR: Application '{application_id}' not found."
    
    app = store.applications[application_id]
    applicant = store.applicants.get(app.applicant_id)
    credit = store.credit_reports.get(app.applicant_id)
    
    if not applicant:
        return "ERROR: Applicant not found."
    
    if not credit:
        return f"ERROR: No credit report on file. Run `check_credit_score(applicant_id='{app.applicant_id}')` first."
    
    # Calculate monthly income
    monthly_income = applicant.annual_income / 12
    
    # Estimate monthly payment for DTI
    estimated_rate = get_base_interest_rate(app.loan_type, credit.credit_score)
    estimated_payment = calculate_monthly_payment(
        app.requested_amount, estimated_rate, app.loan_term_months
    )
    
    # Calculate DTI
    dti = calculate_dti_ratio(monthly_income, applicant.monthly_expenses, estimated_payment)
    
    # Loan to income ratio
    lti = app.requested_amount / applicant.annual_income
    
    # Assess risk
    risk_score, risk_level = assess_risk(
        credit.credit_score, dti, applicant.years_employed, lti
    )
    
    # Update application
    app.risk_score = risk_score
    app.risk_level = risk_level.value
    app.status = LoanStatus.UNDER_REVIEW.value
    app.updated_at = datetime.now().isoformat()
    
    output = f"## 📊 Risk Assessment: {application_id}\n\n"
    
    # Risk score visualization
    if risk_level == RiskLevel.LOW:
        output += f"**Risk Score**: {risk_score}/100 🟢 LOW RISK\n\n"
    elif risk_level == RiskLevel.MODERATE:
        output += f"**Risk Score**: {risk_score}/100 🟡 MODERATE RISK\n\n"
    elif risk_level == RiskLevel.HIGH:
        output += f"**Risk Score**: {risk_score}/100 🟠 HIGH RISK\n\n"
    else:
        output += f"**Risk Score**: {risk_score}/100 🔴 VERY HIGH RISK\n\n"
    
    output += "### Risk Factors Analysis\n\n"
    
    output += f"**Credit Score**: {credit.credit_score}\n"
    if credit.credit_score >= 750:
        output += "  ✅ Excellent - Prime borrower\n"
    elif credit.credit_score >= 700:
        output += "  ✅ Good - Qualified borrower\n"
    elif credit.credit_score >= 650:
        output += "  ⚠️ Fair - Subprime territory\n"
    else:
        output += "  ❌ Poor - High default risk\n"
    
    output += f"\n**Debt-to-Income Ratio**: {dti:.1f}%\n"
    if dti <= 30:
        output += "  ✅ Low debt burden\n"
    elif dti <= 40:
        output += "  ⚠️ Moderate debt burden\n"
    else:
        output += "  ❌ High debt burden\n"
    
    output += f"\n**Employment Stability**: {applicant.years_employed} years\n"
    if applicant.years_employed >= 3:
        output += "  ✅ Stable employment history\n"
    elif applicant.years_employed >= 1:
        output += "  ⚠️ Relatively new employment\n"
    else:
        output += "  ❌ Limited employment history\n"
    
    output += f"\n**Loan-to-Income Ratio**: {lti:.2f}x annual income\n"
    if lti <= 2:
        output += "  ✅ Conservative loan amount\n"
    elif lti <= 4:
        output += "  ⚠️ Moderate loan amount\n"
    else:
        output += "  ❌ Large loan relative to income\n"
    
    output += "\n### Recommendation\n"
    if risk_level == RiskLevel.LOW:
        output += "✅ **APPROVE** - Low risk, standard terms recommended\n"
    elif risk_level == RiskLevel.MODERATE:
        output += "⚠️ **CONDITIONAL APPROVE** - Consider higher rate or reduced amount\n"
    elif risk_level == RiskLevel.HIGH:
        output += "🟠 **REVIEW REQUIRED** - Requires additional documentation/collateral\n"
    else:
        output += "❌ **DECLINE RECOMMENDED** - Risk exceeds acceptable threshold\n"
    
    return output


@tool
def calculate_interest_rate(application_id: str) -> str:
    """Calculate the interest rate for a loan application.
    
    Based on loan type, credit score, and risk assessment.
    
    Args:
        application_id: The loan application ID
        
    Returns:
        Interest rate and payment details
    """
    store = get_data_store()
    
    if application_id not in store.applications:
        return f"ERROR: Application '{application_id}' not found."
    
    app = store.applications[application_id]
    applicant = store.applicants.get(app.applicant_id)
    credit = store.credit_reports.get(app.applicant_id)
    
    if not credit:
        return f"ERROR: No credit report. Run check_credit_score first."
    
    # Get base rate
    base_rate = get_base_interest_rate(app.loan_type, credit.credit_score)
    
    # Adjust for risk (if assessed)
    if app.risk_score is not None:
        if app.risk_score < 60:
            base_rate += 2.0
        elif app.risk_score < 70:
            base_rate += 1.0
    
    # Adjust for collateral
    if app.collateral_value and app.collateral_value >= app.requested_amount * 0.8:
        base_rate -= 0.5
    
    # Cap the rate
    final_rate = max(4.0, min(base_rate, 25.0))
    
    # Calculate payments
    monthly_payment = calculate_monthly_payment(
        app.requested_amount, final_rate, app.loan_term_months
    )
    total_interest = calculate_total_interest(
        app.requested_amount, monthly_payment, app.loan_term_months
    )
    total_cost = app.requested_amount + total_interest
    
    # Update application
    app.interest_rate = final_rate
    app.monthly_payment = monthly_payment
    app.updated_at = datetime.now().isoformat()
    
    output = f"## 💰 Interest Rate Calculation: {application_id}\n\n"
    output += f"### Loan Terms\n"
    output += f"- **Principal**: ${app.requested_amount:,.2f}\n"
    output += f"- **Term**: {app.loan_term_months} months ({app.loan_term_months//12} years)\n"
    output += f"- **Interest Rate**: **{final_rate:.2f}% APR**\n\n"
    
    output += f"### Payment Details\n"
    output += f"- **Monthly Payment**: **${monthly_payment:,.2f}**\n"
    output += f"- **Total Interest**: ${total_interest:,.2f}\n"
    output += f"- **Total Cost**: ${total_cost:,.2f}\n\n"
    
    output += f"### Rate Breakdown\n"
    output += f"- Base rate for {app.loan_type}: {get_base_interest_rate(app.loan_type, 700):.1f}%\n"
    output += f"- Credit score adjustment: {final_rate - get_base_interest_rate(app.loan_type, 700):+.1f}%\n"
    
    if app.collateral_value:
        output += f"- Collateral discount: -0.5%\n"
    
    output += f"\n**Final Rate**: {final_rate:.2f}% APR\n"
    
    return output


@tool
def process_loan_decision(application_id: str, override_decision: str = None) -> str:
    """Process final approval/rejection decision for a loan.
    
    Args:
        application_id: The loan application ID
        override_decision: Optional manual override ('approve' or 'reject')
        
    Returns:
        Decision with reasoning
    """
    store = get_data_store()
    
    if application_id not in store.applications:
        return f"ERROR: Application '{application_id}' not found."
    
    app = store.applications[application_id]
    applicant = store.applicants.get(app.applicant_id)
    credit = store.credit_reports.get(app.applicant_id)
    
    if app.status in [LoanStatus.APPROVED.value, LoanStatus.REJECTED.value]:
        return f"Application already processed. Status: {app.status}"
    
    if not app.risk_score:
        return f"ERROR: Risk assessment not done. Run assess_loan_risk first."
    
    if not app.interest_rate:
        return f"ERROR: Interest rate not calculated. Run calculate_interest_rate first."
    
    # Make decision
    if override_decision:
        approved = override_decision.lower() == 'approve'
    else:
        # Auto-decision based on risk
        approved = app.risk_score >= 50
    
    if approved:
        app.status = LoanStatus.APPROVED.value
        app.approved_at = datetime.now().isoformat()
        
        reasons = []
        if credit.credit_score >= 700:
            reasons.append(f"Good credit score ({credit.credit_score})")
        if app.risk_score >= 70:
            reasons.append(f"Low risk profile (score: {app.risk_score})")
        if applicant.years_employed >= 3:
            reasons.append("Stable employment history")
        
        app.decision_reason = "APPROVED: " + ", ".join(reasons) if reasons else "APPROVED: Meets lending criteria"
        
        output = f"## ✅ LOAN APPROVED: {application_id}\n\n"
        output += f"Congratulations! The loan has been approved.\n\n"
        output += f"### Approved Terms\n"
        output += f"- **Amount**: ${app.requested_amount:,.2f}\n"
        output += f"- **Interest Rate**: {app.interest_rate:.2f}% APR\n"
        output += f"- **Monthly Payment**: ${app.monthly_payment:,.2f}\n"
        output += f"- **Term**: {app.loan_term_months} months\n\n"
        output += f"### Approval Reasons\n"
        for reason in reasons:
            output += f"- {reason}\n"
        
    else:
        app.status = LoanStatus.REJECTED.value
        
        reasons = []
        if credit.credit_score < 600:
            reasons.append(f"Credit score below minimum ({credit.credit_score})")
        if app.risk_score < 50:
            reasons.append(f"High risk profile (score: {app.risk_score})")
        if credit.derogatory_marks > 0:
            reasons.append(f"{credit.derogatory_marks} derogatory marks on credit")
        
        app.decision_reason = "REJECTED: " + ", ".join(reasons) if reasons else "REJECTED: Does not meet lending criteria"
        
        output = f"## ❌ LOAN REJECTED: {application_id}\n\n"
        output += f"Unfortunately, we cannot approve this loan at this time.\n\n"
        output += f"### Rejection Reasons\n"
        for reason in reasons:
            output += f"- {reason}\n"
        output += f"\n### Recommendations\n"
        output += "- Improve credit score before reapplying\n"
        output += "- Consider a smaller loan amount\n"
        output += "- Add a co-signer with better credit\n"
        output += "- Provide collateral to reduce risk\n"
    
    app.updated_at = datetime.now().isoformat()
    
    return output


@tool
def generate_payment_schedule(application_id: str, num_payments: int = 12) -> str:
    """Generate amortization/payment schedule for an approved loan.
    
    Args:
        application_id: The loan application ID
        num_payments: Number of payments to show (default: 12)
        
    Returns:
        Payment schedule table
    """
    store = get_data_store()
    
    if application_id not in store.applications:
        return f"ERROR: Application '{application_id}' not found."
    
    app = store.applications[application_id]
    
    if app.status != LoanStatus.APPROVED.value:
        return f"ERROR: Loan must be approved first. Current status: {app.status}"
    
    if not app.interest_rate:
        return "ERROR: Interest rate not set."
    
    schedule = generate_amortization_schedule(
        app.requested_amount,
        app.interest_rate,
        app.loan_term_months
    )
    
    output = f"## 📅 Payment Schedule: {application_id}\n\n"
    output += f"**Principal**: ${app.requested_amount:,.2f}\n"
    output += f"**Rate**: {app.interest_rate:.2f}% APR\n"
    output += f"**Monthly Payment**: ${app.monthly_payment:,.2f}\n\n"
    
    output += "| # | Due Date | Principal | Interest | Payment | Balance |\n"
    output += "|---|---|---|---|---|---|\n"
    
    for payment in schedule[:num_payments]:
        output += f"| {payment.payment_number} | {payment.due_date} | "
        output += f"${payment.principal:,.2f} | ${payment.interest:,.2f} | "
        output += f"${payment.total_payment:,.2f} | ${payment.remaining_balance:,.2f} |\n"
    
    if len(schedule) > num_payments:
        output += f"\n*Showing {num_payments} of {len(schedule)} payments*\n"
    
    # Summary
    total_interest = sum(p.interest for p in schedule)
    total_paid = sum(p.total_payment for p in schedule)
    
    output += f"\n### Loan Summary\n"
    output += f"- Total Payments: {len(schedule)}\n"
    output += f"- Total Interest: ${total_interest:,.2f}\n"
    output += f"- Total Amount Paid: ${total_paid:,.2f}\n"
    
    return output


@tool
def calculate_loan_affordability(
    annual_income: float,
    monthly_expenses: float,
    interest_rate: float = 7.0,
    term_months: int = 60
) -> str:
    """Calculate how much loan an applicant can afford.
    
    Based on DTI limits and income.
    
    Args:
        annual_income: Annual gross income
        monthly_expenses: Current monthly expenses
        interest_rate: Expected interest rate (default: 7%)
        term_months: Loan term in months (default: 60)
        
    Returns:
        Affordability analysis
    """
    monthly_income = annual_income / 12
    
    # Calculate max payment at different DTI thresholds
    dti_limits = [
        (30, "Conservative"),
        (36, "Standard"),
        (43, "Maximum (FHA)")
    ]
    
    output = "## 💵 Loan Affordability Analysis\n\n"
    output += f"### Input Parameters\n"
    output += f"- Annual Income: ${annual_income:,.2f}\n"
    output += f"- Monthly Income: ${monthly_income:,.2f}\n"
    output += f"- Current Expenses: ${monthly_expenses:,.2f}\n"
    output += f"- Interest Rate: {interest_rate}%\n"
    output += f"- Term: {term_months} months\n\n"
    
    output += "### Maximum Loan Amounts by DTI\n\n"
    output += "| DTI Level | Max Payment | Max Loan | Monthly After |\n"
    output += "|---|---|---|---|\n"
    
    for dti_limit, label in dti_limits:
        max_total_debt = monthly_income * (dti_limit / 100)
        max_new_payment = max_total_debt - monthly_expenses
        
        if max_new_payment <= 0:
            output += f"| {label} ({dti_limit}%) | N/A | N/A | N/A |\n"
            continue
        
        # Reverse calculate max principal from payment
        monthly_rate = interest_rate / 12 / 100
        if monthly_rate > 0:
            max_principal = max_new_payment * \
                           ((1 - (1 + monthly_rate)**(-term_months)) / monthly_rate)
        else:
            max_principal = max_new_payment * term_months
        
        remaining = monthly_income - monthly_expenses - max_new_payment
        
        output += f"| {label} ({dti_limit}%) | ${max_new_payment:,.0f} | "
        output += f"${max_principal:,.0f} | ${remaining:,.0f} |\n"
    
    output += "\n### Recommendations\n"
    output += "- **Conservative (30% DTI)**: Best for financial flexibility\n"
    output += "- **Standard (36% DTI)**: Common lending threshold\n"
    output += "- **Maximum (43% DTI)**: Absolute maximum for most lenders\n"
    
    return output


@tool
def compare_loan_options(
    principal: float,
    options: str = "standard"
) -> str:
    """Compare different loan term and rate options.
    
    Args:
        principal: Loan amount
        options: 'standard' for common options or 'all' for comprehensive
        
    Returns:
        Comparison table
    """
    if options == "all":
        terms = [12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 360]
    else:
        terms = [24, 36, 60, 84, 120]
    
    rates = [5.0, 6.0, 7.0, 8.0, 10.0]
    
    output = f"## 📊 Loan Comparison: ${principal:,.2f}\n\n"
    
    output += "### Monthly Payments by Term and Rate\n\n"
    output += "| Term | " + " | ".join([f"{r}% APR" for r in rates]) + " |\n"
    output += "|---" + "|---" * len(rates) + "|\n"
    
    for term in terms:
        years = f"{term} mo ({term//12}y)" if term >= 12 else f"{term} mo"
        row = f"| {years} |"
        for rate in rates:
            payment = calculate_monthly_payment(principal, rate, term)
            row += f" ${payment:,.0f} |"
        output += row + "\n"
    
    output += "\n### Total Interest Comparison\n\n"
    output += "| Term | " + " | ".join([f"{r}% APR" for r in rates]) + " |\n"
    output += "|---" + "|---" * len(rates) + "|\n"
    
    for term in terms[:5]:  # Limit for readability
        years = f"{term} mo"
        row = f"| {years} |"
        for rate in rates:
            payment = calculate_monthly_payment(principal, rate, term)
            total_interest = calculate_total_interest(principal, payment, term)
            row += f" ${total_interest:,.0f} |"
        output += row + "\n"
    
    output += "\n### Key Insights\n"
    output += "- **Shorter terms**: Higher payments but less total interest\n"
    output += "- **Longer terms**: Lower payments but more interest paid\n"
    output += "- **Lower rates**: Always save money - improve credit before applying\n"
    
    return output


# =============================================================================
# Loan Agent System Prompt
# =============================================================================

LOAN_SYSTEM_PROMPT = """You are the Loan Application Agent - an AI-powered loan processor that helps users with loan applications, credit assessment, and financial decisions.

## Your Capabilities

### Applicant Management
- `create_applicant` - Create new applicant profile
- `get_applicant_info` - View applicant details
- `list_applicants` - List all applicants
- `check_credit_score` - Pull credit report

### Loan Applications
- `submit_loan_application` - Submit new loan application
- `get_application_status` - Check application status
- `list_applications` - List all applications

### Risk & Decisioning
- `assess_loan_risk` - Perform risk assessment
- `calculate_interest_rate` - Calculate loan rates
- `process_loan_decision` - Approve or reject loan

### Financial Tools
- `generate_payment_schedule` - Create amortization schedule
- `calculate_loan_affordability` - Determine how much can be borrowed
- `compare_loan_options` - Compare different loan terms

## Loan Types Supported
- **Personal**: Unsecured personal loans
- **Mortgage**: Home purchase/refinance
- **Auto**: Vehicle financing
- **Business**: Small business loans
- **Student**: Education financing
- **Home Equity**: HELOC/home equity loans

## Typical Workflow

1. **Create Applicant**: Collect applicant information
2. **Credit Check**: Pull credit score and report
3. **Submit Application**: Create loan application
4. **Risk Assessment**: Evaluate loan risk
5. **Calculate Rate**: Determine interest rate
6. **Decision**: Approve or reject
7. **Generate Schedule**: Create payment plan

## Decision Criteria

**Approval Factors**:
- Credit score ≥ 650
- DTI ratio ≤ 43%
- Stable employment (2+ years)
- Risk score ≥ 50

**Rejection Factors**:
- Credit score < 580
- DTI ratio > 50%
- Recent bankruptcies
- Insufficient income

## Sample Data

The system comes with sample applicants:
- APP-001: John Smith (Good credit, employed)
- APP-002: Jane Doe (Fair credit, self-employed)

And a sample application:
- LA-2024-001: $25,000 personal loan

Be helpful, accurate, and guide users through the loan process!"""


# =============================================================================
# All Loan Tools
# =============================================================================

LOAN_TOOLS = [
    # Applicant Management
    create_applicant,
    get_applicant_info,
    list_applicants,
    check_credit_score,
    
    # Applications
    submit_loan_application,
    get_application_status,
    list_applications,
    
    # Risk & Decision
    assess_loan_risk,
    calculate_interest_rate,
    process_loan_decision,
    
    # Financial Tools
    generate_payment_schedule,
    calculate_loan_affordability,
    compare_loan_options,
]


# =============================================================================
# Middleware
# =============================================================================

@wrap_tool_call
def handle_loan_errors(request, handler):
    """Handle tool execution errors."""
    try:
        return handler(request)
    except Exception as e:
        import traceback
        error_msg = f"Loan tool error: {type(e).__name__}: {e}"
        print(f"[LOAN ERROR] {error_msg}")
        traceback.print_exc()
        return ToolMessage(
            content=error_msg,
            tool_call_id=request.tool_call["id"]
        )


# =============================================================================
# Agent Creation
# =============================================================================

def create_loan_agent(verbose: bool = True):
    """Create the Loan Application agent.
    
    Args:
        verbose: Enable verbose output
        
    Returns:
        Configured loan agent
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable required")
    
    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.1,
        max_tokens=4096,
    )
    
    agent = create_agent(
        model=model,
        tools=LOAN_TOOLS,
        system_prompt=LOAN_SYSTEM_PROMPT,
        middleware=[handle_loan_errors],
    )
    
    return agent


def extract_response(result) -> str:
    """Extract text response from agent result."""
    if not isinstance(result, dict):
        return str(result)
    
    messages = result.get("messages", [])
    if not messages:
        return str(result)
    
    last_msg = messages[-1]
    content = getattr(last_msg, 'content', None)
    
    if content is None:
        return str(last_msg)
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get('type') == 'text':
                text_parts.append(block.get('text', ''))
            elif hasattr(block, 'text'):
                text_parts.append(block.text)
        return '\n'.join(text_parts) if text_parts else str(content)
    
    return str(content)


# =============================================================================
# Interactive Mode
# =============================================================================

def run_interactive():
    """Run loan agent in interactive mode."""
    agent = create_loan_agent(verbose=True)
    history = []
    
    print("\n" + "="*60)
    print("  💰 Loan Application Agent")
    print("="*60)
    print("\nI can help you with loan applications and financial decisions.")
    print("\nTry saying:")
    print("  • 'Show me the applicants'")
    print("  • 'Check credit score for APP-001'")
    print("  • 'Submit a $50,000 mortgage application for APP-001'")
    print("  • 'Process loan decision for LA-2024-001'")
    print("  • 'How much can I afford with $80,000 income?'")
    print("  • 'Compare loan options for $30,000'")
    print("\nType 'quit' to exit.\n")
    sys.stdout.flush()
    
    while True:
        try:
            sys.stdout.write("\033[92mYou:\033[0m ")
            sys.stdout.flush()
            user_input = sys.stdin.readline()
            
            if not user_input:
                print("\nEnd of input.")
                break
            
            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            messages = history + [{"role": "user", "content": user_input}]
            
            print("\n\033[96m💰 Loan Agent:\033[0m ", end="")
            sys.stdout.flush()
            
            # Stream response
            full_response = ""
            tool_calls_seen = set()
            
            for chunk in agent.stream(
                {"messages": messages},
                stream_mode="values",
                config={"recursion_limit": 100}
            ):
                if "messages" in chunk and chunk["messages"]:
                    latest_message = chunk["messages"][-1]
                    
                    content = getattr(latest_message, 'content', '')
                    if isinstance(content, str) and content:
                        new_content = content[len(full_response):]
                        if new_content:
                            print(new_content, end="", flush=True)
                            full_response = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text = block.get('text', '')
                                new_content = text[len(full_response):]
                                if new_content:
                                    print(new_content, end="", flush=True)
                                    full_response = text
                    
                    tool_calls = getattr(latest_message, 'tool_calls', None)
                    if tool_calls:
                        for tc in tool_calls:
                            tc_id = tc.get('id', '')
                            if tc_id and tc_id not in tool_calls_seen:
                                tool_calls_seen.add(tc_id)
                                print(f"\n\n🔧 **{tc.get('name', 'unknown')}**", flush=True)
            
            print("\n")
            sys.stdout.flush()
            
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": full_response})
            
            if len(history) > 20:
                history = history[-20:]
                
        except EOFError:
            print("\nEnd of input.")
            break
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()
    
    print("\n👋 Goodbye!")


def run_single_message(message: str):
    """Process a single message and exit."""
    agent = create_loan_agent(verbose=True)
    
    print(f"\n💰 Processing: {message}\n")
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"recursion_limit": 100}
    )
    
    response = extract_response(result)
    print(response)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Loan Application Agent")
    parser.add_argument("--auto", type=str, help="Process a single message")
    parser.add_argument("message", nargs="?", help="Message to process (with --auto)")
    
    args = parser.parse_args()
    
    if args.auto:
        run_single_message(args.auto)
    elif args.message:
        run_single_message(args.message)
    else:
        run_interactive()


if __name__ == "__main__":
    main()