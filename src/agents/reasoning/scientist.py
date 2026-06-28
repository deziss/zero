"""src/agents/reasoning/scientist.py — Scientist agent for scientific deduction

The Scientist agent:
  - Generates hypotheses about system behavior
  - Designs experiments to test hypotheses
  - Builds evidence-based theories
  - Refines theories when evidence contradicts
  - Proposes novel scientific questions
"""

import json
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class Confidence:
    """Confidence levels for scientific claims."""
    HIGH = 0.8
    MEDIUM = 0.5
    LOW = 0.2
    NULL = 0.0


class HypothesisCategory(Enum):
    MECHANISTIC = "mechanistic"      # "X causes Y through Z"
    CORRELATIONAL = "correlational"  # "X is correlated with Y"
    PREDICTIVE = "predictive"        # "If X, then Y"
    EXPLANATORY = "explanatory"      # "Y occurs because of Z"
    FALSIFICATION = "falsification"  # "X cannot be explained by Y"


@dataclass
class ScientistAgent:
    """
    Agent that performs scientific reasoning.
    
    Capabilities:
    1. Generates testable hypotheses from observations
    2. Designs experiments to evaluate hypotheses
    3. Analyzes experimental results
    4. Updates scientific confidence
    5. Proposes new research directions
    """
    
    name: str = "Scientist"
    expertise: str = "general_science"
    n_memories: int = 100
    exploration_rate: float = 0.3
    
    beliefs: Dict[str, float] = field(default_factory=dict)
    memories: List[Dict] = field(default_factory=list)
    active_hypotheses: List[Dict] = field(default_factory=list)
    
    def deduce_hypothesis(
        self,
        observations: List[Dict],
        context: Optional[Dict] = None,
    ) -> Dict:
        """Deduce a hypothesis from observations."""
        # TODO: Use pattern recognition to deduce hypothesis
        # For now, create from observations
        if not observations:
            return {'hypothesis': '', 'confidence': 0, 'category': None}
        
        # Extract patterns
        n_patterns = len(set(
            tuple(o.get('key', '') for o in observations)
        ))
        confidence = min(0.9, 0.3 * n_patterns)
        
        return {
            'hypothesis': f'When {observations[0].get("key", "X")} is high, {observations[0].get("value", "Y")} tends to increase',
            'confidence': confidence,
            'category': 'correlational',
            'evidence_n': len(observations),
        }
    
    def design_experiment(
        self,
        hypothesis: Dict,
        available_tools: Optional[List[str]] = None,
    ) -> Dict:
        """Design experiment to test a hypothesis."""
        tools = available_tools or ['simulation', 'observation', 'intervention']
        
        return {
            'test_type': random.choice(tools),
            'sample_size': random.randint(10, 100),
            'duration_steps': random.randint(100, 1000),
            'metrics': ['accuracy', 'precision', 'recall'],
            'null_hypothesis': f'No relationship between {hypothesis.get("evidence_n", 0)} observations',
        }
    
    def evaluate_result(
        self,
        result: Dict,
        hypothesis: Optional[Dict] = None,
    ) -> Dict:
        """Evaluate experimental result against hypothesis."""
        score = result.get('score', 0)
        
        if score > 0.7:
            verdict = 'supported'
            confidence_change = 0.1
        elif score > 0.4:
            verdict = 'inconclusive'
            confidence_change = -0.05
        else:
            verdict = 'refuted'
            confidence_change = -0.15
        
        return {
            'verdict': verdict,
            'confidence_change': confidence_change,
            'score': score,
            'strength': score if score > 0.5 else 1 - score,
        }
    
    def update_belief(self, key: str, new_score: float, weight: float = 0.1):
        """Update belief confidence using Bayesian update."""
        old_belief = self.beliefs.get(key, Confidence.NULL)
        self.beliefs[key] = old_belief * (1 - weight) + new_score * weight
        
        if self.beliefs[key] > Confidence.LOW:
            self.active_hypotheses.append({
                'key': key,
                'confidence': self.beliefs[key],
                'updated_at': time.time(),
            })
    
    def propose_new_query(self, current_queries: List[str]) -> str:
        """Propose a new scientific question."""
        explored = set(current_queries)
        unexplored = [
            f"Does {x} affect {y}?",
            f"What is the relationship between {x} and {y}?",
            f"Can we predict {y} from {x}?",
        ]
        for q in unexplored:
            if q not in explored:
                return q
        return "What happens if we vary all parameters simultaneously?"
    
    def summarize(self) -> str:
        """Summarize agent's current state."""
        strong_beliefs = [k for k, v in self.beliefs.items() if v > Confidence.HIGH]
        medium_beliefs = [k for k, v in self.beliefs.items() if Confidence.MEDIUM < v <= Confidence.HIGH]
        weak_beliefs = [k for k, v in self.beliefs.items() if Confidence.LOW < v <= Confidence.MEDIUM]
        
        return (
            f"Agent: {self.name}\n"
            f"Strong beliefs: {len(strong_beliefs)}\n"
            f"Medium beliefs: {len(medium_beliefs)}\n"
            f"Weak beliefs: {len(weak_beliefs)}\n"
            f"Active hypotheses: {len(self.active_hypotheses)}\n"
            f"Memories: {len(self.memories)}"
        )


@dataclass
class HypothesisTest:
    """A single hypothesis test."""
    id: str
    hypothesis: str
    test_type: str
    result: float
    confidence: float
    n_samples: int
    
    def is_supported(self, threshold: float = 0.6) -> bool:
        return self.result > threshold
    
    def strength(self) -> str:
        if self.result > 0.8:
            return "strong"
        elif self.result > 0.5:
            return "moderate"
        elif self.result > 0.3:
            return "weak"
        return "very weak or negative"

@dataclass
class TheoryBuilder:
    """
    Constructs theories from supported hypotheses.
    
    Builds theories by:
    1. Collecting supporting hypotheses
    2. Finding common mechanisms
    3. Forming general principles
    4. Assigning confidence based on evidence strength
    """
    
    theories: List[TheoryBuilder] = field(default_factory=list)
    confidence: float = 0.0
    statements: List[str] = field(default_factory=list)
    supporting_tests: List[Dict] = field(default_factory=list)
    falsifying_tests: List[Dict] = field(default_factory=list)
    
    def add_hypothesis(self, test: HypothesisTest):
        """Add a hypothesis test to the theory."""
        if test.is_supporting():
            self.supporting_tests.append(test.__dict__)
        else:
            self.falsifying_tests.append(test.__dict__)
        
        self.confidence = (
            len(self.supporting_tests) /
            max(len(self.supporting_tests) + len(self.falsifying_tests), 1)
        )
        
        self._update_statements()
    
    def _update_statements(self):
        """Update theory statements from supporting tests."""
        self.statements = [
            f"Observation: {t['hypothesis']} (strength={t['strength']})"
            for t in self.supporting_tests
        ]
        
        # Summarize into common principle
        if len(self.statements) >= 3:
            # Find common patterns
            words = []
            for stmt in self.statements:
                words.extend(stmt.split())
            
            freq = {}
            for w in words:
                freq[w] = freq.get(w, 0) + 1
            
            common = sorted(freq.items(), key=lambda x: -x[1])[:5]
            self.statements.append(
                f"Common theme across {len(self.statements)} observations: {', '.join(w for w, _ in common)}"
            )
    
    def confidence_category(self) -> str:
        if self.confidence > 0.8:
            return "well-established"
        elif self.confidence > 0.5:
            return "supported-by-evidence"
        elif self.confidence > 0.3:
            return "tentative"
        else:
            return "weak"
    
    def summary(self) -> str:
        summary_str = (f"Theory: {self.__class__.__name__}\n"
                       f"Confidence: {self.confidence:.2f} ({self.confidence_category()})\n"
                       f"Supporting tests: {len(self.supporting_tests)}\n"
                       f"Falsifying tests: {len(self.falsifying_tests)}\n"
                       f"Statements ({len(self.statements)}):\n")
        for s in self.statements:
            summary_str += f'  {s}\n'
        return summary_str
