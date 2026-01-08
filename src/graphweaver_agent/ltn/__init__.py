"""LTN (Logic Tensor Networks) module for learning and generating business rules.

FIXED: Lazy loading of TensorFlow/LTN to avoid conflicts with sentence-transformers.
TensorFlow is only imported when LTN functionality is actually used.
"""

from .predicates import (
    LTNPredicateFactory,
    TablePredicate,
    ColumnPredicate,
    FKPredicate,
    RelationshipPredicate,
)
from .rule_learner import (
    LTNRuleLearner,
    LearnedRule,
    RuleLearningConfig,
)
from .rule_generator import (
    BusinessRuleGenerator,
    GeneratedRule,
    RuleTemplate,
)
from .knowledge_base import (
    LTNKnowledgeBase,
    Axiom,
    Constraint,
    is_ltn_available,
)
from .trainer import (
    LTNTrainer,
    TrainingConfig,
    TrainingResult,
)

__all__ = [
    "LTNPredicateFactory",
    "TablePredicate",
    "ColumnPredicate", 
    "FKPredicate",
    "RelationshipPredicate",
    "LTNRuleLearner",
    "LearnedRule",
    "RuleLearningConfig",
    "BusinessRuleGenerator",
    "GeneratedRule",
    "RuleTemplate",
    "LTNKnowledgeBase",
    "Axiom",
    "Constraint",
    "LTNTrainer",
    "TrainingConfig",
    "TrainingResult",
    "is_ltn_available",
]
