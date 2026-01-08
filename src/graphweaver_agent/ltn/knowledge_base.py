"""
LTN Knowledge Base - Stores axioms, constraints, and predicates.

The knowledge base contains:
- Axioms: Facts that are known to be true
- Constraints: Rules that should be satisfied
- Queries: Logical questions to answer

FIXED: Lazy loading of TensorFlow/LTN to avoid conflicts with sentence-transformers.
"""
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Lazy-loaded modules
_ltn = None
_tf = None
_LTN_AVAILABLE = None


def _ensure_ltn_loaded():
    """Lazy load LTN and TensorFlow only when needed."""
    global _ltn, _tf, _LTN_AVAILABLE
    
    if _LTN_AVAILABLE is None:
        try:
            import ltn
            import tensorflow as tf
            _ltn = ltn
            _tf = tf
            _LTN_AVAILABLE = True
        except ImportError:
            _LTN_AVAILABLE = False
            _ltn = None
            _tf = None
    
    return _LTN_AVAILABLE


def get_ltn():
    """Get the ltn module (lazy loaded)."""
    _ensure_ltn_loaded()
    if _ltn is None:
        raise ImportError("LTN not installed. Run: pip install ltn tensorflow")
    return _ltn


def get_tf():
    """Get the tensorflow module (lazy loaded)."""
    _ensure_ltn_loaded()
    if _tf is None:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
    return _tf


def is_ltn_available() -> bool:
    """Check if LTN is available without triggering import."""
    return _ensure_ltn_loaded()


class AxiomType(Enum):
    """Types of axioms."""
    FACT = "fact"  # Ground truth from data
    RULE = "rule"  # Logical implication
    CONSTRAINT = "constraint"  # Must be satisfied
    SOFT_CONSTRAINT = "soft_constraint"  # Should be satisfied


@dataclass
class Axiom:
    """A logical axiom in the knowledge base."""
    name: str
    formula: str  # Human-readable formula
    axiom_type: AxiomType
    weight: float = 1.0
    description: str = ""
    ltn_formula: Any = None  # LTN formula object
    
    def __str__(self):
        return f"{self.name}: {self.formula}"


@dataclass
class Constraint:
    """A constraint that must be satisfied."""
    name: str
    formula: str
    penalty: float = 1.0
    is_hard: bool = False
    description: str = ""


class LTNKnowledgeBase:
    """Knowledge base for LTN reasoning."""
    
    def __init__(self):
        self.axioms: Dict[str, Axiom] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.constants: Dict[str, Any] = {}
        self.variables: Dict[str, Any] = {}
        self.predicates: Dict[str, Any] = {}
        self.functions: Dict[str, Any] = {}
        
        # Fuzzy operators - lazy initialized
        self._operators_initialized = False
        self._Not = None
        self._And = None
        self._Or = None
        self._Implies = None
        self._Equiv = None
        self._Forall = None
        self._Exists = None
    
    def _init_operators(self):
        """Initialize LTN fuzzy operators (lazy)."""
        if self._operators_initialized:
            return
        
        if not _ensure_ltn_loaded():
            return
        
        ltn = get_ltn()
        self._Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
        self._And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
        self._Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
        self._Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
        self._Equiv = ltn.Wrapper_Connective(ltn.fuzzy_ops.Equiv(
            and_op=ltn.fuzzy_ops.And_Prod(), 
            implies_op=ltn.fuzzy_ops.Implies_Reichenbach()
        ))
        self._Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall")
        self._Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=2), semantics="exists")
        self._operators_initialized = True
    
    @property
    def Not(self):
        self._init_operators()
        return self._Not
    
    @property
    def And(self):
        self._init_operators()
        return self._And
    
    @property
    def Or(self):
        self._init_operators()
        return self._Or
    
    @property
    def Implies(self):
        self._init_operators()
        return self._Implies
    
    @property
    def Equiv(self):
        self._init_operators()
        return self._Equiv
    
    @property
    def Forall(self):
        self._init_operators()
        return self._Forall
    
    @property
    def Exists(self):
        self._init_operators()
        return self._Exists
    
    def add_constant(self, name: str, value: np.ndarray) -> Any:
        """Add a constant (grounded entity)."""
        ltn = get_ltn()
        const = ltn.Constant(value, trainable=False)
        self.constants[name] = const
        return const
    
    def add_variable(self, name: str, values: np.ndarray) -> Any:
        """Add a variable (set of entities)."""
        ltn = get_ltn()
        var = ltn.Variable(name, values)
        self.variables[name] = var
        return var
    
    def add_predicate(self, name: str, predicate: Any):
        """Add a predicate."""
        self.predicates[name] = predicate
    
    def add_axiom(self, axiom: Axiom):
        """Add an axiom to the knowledge base."""
        self.axioms[axiom.name] = axiom
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint."""
        self.constraints[constraint.name] = constraint
    
    def add_fk_axioms(self):
        """Add standard FK-related axioms."""
        axioms = [
            Axiom(
                name="pk_unique",
                formula="∀x(IsPK(x) → IsUnique(x))",
                axiom_type=AxiomType.RULE,
                description="Primary keys are always unique",
            ),
            Axiom(
                name="fk_references_pk",
                formula="∀x,y(FK(x,y) → IsPK(y))",
                axiom_type=AxiomType.RULE,
                description="Foreign keys reference primary keys",
            ),
            Axiom(
                name="fk_same_type",
                formula="∀x,y(FK(x,y) → SameType(x,y))",
                axiom_type=AxiomType.RULE,
                description="FK and PK must have compatible types",
            ),
            Axiom(
                name="dimension_has_pk",
                formula="∀t(IsDimension(t) → ∃c(BelongsTo(c,t) ∧ IsPK(c)))",
                axiom_type=AxiomType.RULE,
                description="Dimension tables have primary keys",
            ),
            Axiom(
                name="fact_has_fks",
                formula="∀t(IsFact(t) → ∃c(BelongsTo(c,t) ∧ IsFK(c)))",
                axiom_type=AxiomType.RULE,
                description="Fact tables have foreign keys",
            ),
            Axiom(
                name="junction_two_fks",
                formula="∀t(IsJunction(t) → ∃c1,c2(BelongsTo(c1,t) ∧ BelongsTo(c2,t) ∧ IsFK(c1) ∧ IsFK(c2) ∧ c1≠c2))",
                axiom_type=AxiomType.RULE,
                description="Junction tables have at least two foreign keys",
            ),
        ]
        
        for axiom in axioms:
            self.add_axiom(axiom)
    
    def add_data_quality_axioms(self):
        """Add data quality axioms."""
        axioms = [
            Axiom(
                name="pk_not_null",
                formula="∀x(IsPK(x) → ¬IsNullable(x))",
                axiom_type=AxiomType.CONSTRAINT,
                description="Primary keys cannot be null",
            ),
            Axiom(
                name="fk_referential_integrity",
                formula="∀x,y(FK(x,y) → References(x,y))",
                axiom_type=AxiomType.CONSTRAINT,
                description="FK values must exist in referenced table",
            ),
        ]
        
        for axiom in axioms:
            self.add_axiom(axiom)
    
    def get_all_axioms(self) -> List[Axiom]:
        """Get all axioms."""
        return list(self.axioms.values())
    
    def get_axioms_by_type(self, axiom_type: AxiomType) -> List[Axiom]:
        """Get axioms of a specific type."""
        return [a for a in self.axioms.values() if a.axiom_type == axiom_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export knowledge base to dictionary."""
        return {
            "axioms": [
                {
                    "name": a.name,
                    "formula": a.formula,
                    "type": a.axiom_type.value,
                    "weight": a.weight,
                    "description": a.description,
                }
                for a in self.axioms.values()
            ],
            "constraints": [
                {
                    "name": c.name,
                    "formula": c.formula,
                    "penalty": c.penalty,
                    "is_hard": c.is_hard,
                }
                for c in self.constraints.values()
            ],
            "predicates": list(self.predicates.keys()),
        }
    
    @classmethod
    def create_default(cls) -> "LTNKnowledgeBase":
        """Create knowledge base with default axioms."""
        kb = cls()
        kb.add_fk_axioms()
        kb.add_data_quality_axioms()
        return kb
