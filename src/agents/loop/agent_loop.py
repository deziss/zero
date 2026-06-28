"""src/agents/loop/agent_loop.py — Agent loop for autonomous scientific reasoning

The agent loop manages:
  - State: what the agent is currently doing
  - Planning: generating experiment plans
  - Execution: running simulations/tests
  - Evaluation: analyzing results
  - Learning: updating its model of the system
  - Adaptation: modifying the plan based on results
"""

import json
import time
import uuid
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


class AgentState(Enum):
    """Agent operational states."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    OBSERVING = "observing"
    ANALYZING = "analyzing"
    LEARNING = "learning"
    ADAPTING = "adapting"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Experiment:
    """An experiment proposed by the agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    hypothesis: str = ""
    method: str = ""  # how to test
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict] = None
    status: str = "pending"  # pending, running, done, failed
    score: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'hypothesis': self.hypothesis,
            'method': self.method,
            'params': self.params,
            'result': self.result,
            'status': self.status,
            'score': self.score,
        }

@dataclass
class Theory:
    """A scientific theory the agent is building."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    statements: List[str] = field(default_factory=list)
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    falsifying_evidence: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)
    
    def update_confidence(self, new_score: float):
        """Update theory confidence using Bayesian update."""
        self.confidence = (self.confidence * 0.9) + (new_score * 0.1)
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'statements': self.statements,
            'confidence': self.confidence,
            'evidence_count': len(self.supporting_evidence),
        }

class AgentLoop:
    """
    Central loop for autonomous scientific reasoning.
    
    Manages the cycle:
    Plan → Experiment → Observe → Analyze → Learn → Adapt → Repeat
    
    The agent continuously proposes hypotheses, designs experiments,
    runs simulations, analyzes results, and updates its understanding.
    """
    
    def __init__(
        self,
        goal: str,
        max_iterations: int = 100,
        verbosity: int = 1,
    ):
        self.goal = goal
        self.max_iterations = max_iterations
        self.verbosity = verbosity
        
        self.state = AgentState.IDLE
        self.iteration = 0
        self.theories: List[Theory] = []
        self.experiments: List[Experiment] = []
        self.failed_experiments: List[Experiment] = []
        self.best_result: Optional[Dict] = None
        self.log: List[Dict] = []
        self._running = False
    
    def to_dict(self) -> Dict:
        return {
            'goal': self.goal,
            'state': self.state.value,
            'iteration': self.iteration,
            'n_iterations': self.max_iterations,
            'n_theories': len(self.theories),
            'n_experiments': len(self.experiments),
            'n_failed': len(self.failed_experiments),
            'best_score': self.best_result.get('score', 0) if self.best_result else 0,
        }
    
    def run(self, n_iterations: Optional[int] = None):
        """
        Run the agent loop for the specified number of iterations.
        
        Args:
            n_iterations: number of iterations (defaults to max_iterations)
        """
        n_iter = n_iterations or self.max_iterations
        self._running = True
        self.state = AgentState.PLANNING
        
        for i in range(n_iter):
            if not self._running:
                break
            self.iteration = i + 1
            
            # Step 1: Plan
            self.state = AgentState.PLANNING
            plan = self._plan()
            
            # Step 2: Execute experiments from plan
            self.state = AgentState.EXECUTING
            results = self._execute_plan(plan)
            
            # Step 3: Analyze results
            self.state = AgentState.ANALYZING
            analysis = self._analyze_results(results)
            
            # Step 4: Update theories
            self.state = AgentState.LEARNING
            theory_update = self._update_theories(analysis)
            
            # Step 5: Adapt for next iteration
            self.state = AgentState.ADAPTING
            new_hypotheses = self._adapt(theory_update)
            
            # Log iteration
            self.log.append({
                'iteration': i + 1,
                'state': self.state.value,
                'n_experiments': len(plan),
                'plans': self._plan(),
            })
            
            if self.verbosity >= 2:
                self._log_iteration(i + 1)
            
            if self.iteration >= n_iter:
                self.state = AgentState.COMPLETE
                break
            
            self.state = AgentState.IDLE
            time.sleep(0.1)  # Brief pause (simulates compute)
        
        self._running = False
        return self.to_dict()
    
    def _plan(self) -> List[Experiment]:
        """Generate experiment plan from current knowledge."""
        # TODO: Generate experiments based on:
        # 1. Best theory
        # 2. Most uncertain regions
        # 3. Gap analysis between theories
        pass
    
    def _execute_plan(self, plan: List[Experiment]) -> List[Dict]:
        """Execute experiments and collect results."""
        results = []
        for exp in plan:
            exp.status = "running"
            try:
                result = self._run_experiment(exp)
                exp.result = result
                exp.status = "done"
                exp.score = result.get('score', 0)
                if self.best_result is None or exp.score > self.best_result.get('score', 0):
                    self.best_result = exp.to_dict()
                self.experiments.append(exp)
                results.append(result)
            except Exception as e:
                exp.status = "failed"
                exp.result = {'error': str(e)}
                self.failed_experiments.append(exp)
                results.append({'status': 'failed'})
            time.sleep(0.01)  # Simulate computation
        return results
    
    def _run_experiment(self, experiment: Experiment) -> Dict:
        """Run a single experiment."""
        # TODO: Run actual simulation/physics/physics
        return {'score': 0.5}
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze experiment results."""
        scores = [r.get('score', 0) for r in results]
        return {
            'n_results': len(results),
            'mean_score': sum(scores) / max(len(scores), 1),
            'best_score': max(scores, default=0),
            'results': results,
        }
    
    def _update_theories(self, analysis: Dict) -> List[Theory]:
        """Update/evolve theories based on results."""
        updated = []
        for theory in self.theories:
            # Simulated confidence update
            theory.update_confidence(analysis['mean_score'])
            updated.append(theory)
        return updated
    
    def _adapt(self, theory_update: List[Theory]) -> List[Experiment]:
        """Generate new hypotheses for next iteration."""
        # TODO: Use theory confidence to guide exploration
        return [
            Experiment(
                title=f"Adaptation experiment {self.iteration}",
                hypothesis="New direction from theory adaptation",
                method="adaptive_sampling",
            )
        ]
    
    def _log_iteration(self, iteration: int):
        """Log iteration details."""
        iteration_info = self.log[-1]
        print(f"  [Agent {iteration}] "
              f"experiments={iteration_info.get('n_experiments', 0)}, "
              f"state={iteration_info.get('state', 'unknown')}")
    
    def stop(self):
        """Stop the agent loop."""
        self._running = False
        self.state = AgentState.COMPLETE
    
    def save_checkpoint(self, path: str):
        """Save agent state to file."""
        checkpoint = self.to_dict()
        checkpoint['log'] = self.log
        checkpoint['theories'] = [t.to_dict() for t in self.theories]
        checkpoint['experiments'] = [e.to_dict() for e in self.experiments]
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
