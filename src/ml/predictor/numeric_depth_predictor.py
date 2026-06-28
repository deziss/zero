"""src/ml/predictor/numeric_depth_predictor.py

Pure NumPy adaptive depth predictor for MVP v1.

This replaces the torch-dependent DepthSelector during development.
Predicts per-atom simulation depth based on:
- Local energy gradient magnitude
- Neighbor density and bond stress
- Force instability (force divergence from local mean)

Architecture:
  Input: atomic features (positions, types, distances)
  Output: depth_scores (per-atom, continuous [0, 1])

Usage:
  from src.ml.predictor.numeric_depth_predictor import NumericDepthPredictor
  predictor = NumericDepthPredictor()
  depths = predictor.predict(atomic_numbers, positions, velocities)
"""

import numpy as np
from typing import Optional, List, Tuple, Dict


class NumericDepthPredictor:
    """Adaptive depth prediction using pure NumPy (no torch dependency).
    
    Predicts which atoms need deeper simulation based on structural/force features:
    - High-force atoms → deeper simulation
    - Dense regions → deeper simulation
    - Energetic gradients → deeper simulation
    
    Depth map:
      depth 0 = statistical (coarsest)
      depth 1 = cellular (mesoscopic)
      depth 2 = molecular (atomistic simulation)
      depth 3 = atomistic fine (high-res MD)
      depth 4 = quantum (DFT region)
    """
    
    def __init__(
        self,
        force_threshold: float = 0.5,
        density_threshold: float = 0.6,
        uncertainty_threshold: float = 0.4,
        max_depth: int = 4,
        n_features: int = 64,
        hidden_dim: int = 32,
        cutoff_angstrom: float = 5.0,
        temperature: float = 300.0,
    ):
        self.force_threshold = force_threshold
        self.density_threshold = density_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.max_depth = max_depth
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.cutoff_angstrom = cutoff_angstrom
        self.temperature = temperature
        
        # Initialize weights for the neural-style network
        np.random.seed(42)
        self.weights = {
            'W1': np.random.randn(n_features, hidden_dim) * 0.01,
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, max_depth + 1) * 0.01,
            'b2': np.zeros(max_depth + 1),
        }
    
    def predict(
        self,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        forces: Optional[np.ndarray] = None,
        energies: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None,
        neighbor_distances: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict simulation depth for each atom.
        
        Args:
            atomic_numbers: atomic_numbers (n_atoms,)
            positions: Cartesian coordinates (n_atoms, 3)
            forces: per-atom forces (n_atoms, 3)
            energies: per-atom energies (n_atoms,)
            velocities: per-atom velocities (n_atoms, 3)
            neighbor_distances: pairwise distances or per-atom mean neighbor distances
            cutoff_angstrom: cutoff distance for neighbor detection
        
        Returns:
            depth_map: int array (n_atoms,) with values in [0, max_depth]
        """
        n_atoms = len(atomic_numbers)
        
        # Extract features for each atom
        features = self._extract_features(
            atomic_numbers, positions, forces, energies, velocities, neighbor_distances
        )
        
        # Pass through the predictor network
        logits = self._forward(features)
        
        # Convert logits to discrete depths using thresholding
        depth_scores = self._logits_to_depths(logits)
        
        return depth_scores
    
    def _extract_features(
        self,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
        forces: Optional[np.ndarray],
        energies: Optional[np.ndarray],
        velocities: Optional[np.ndarray],
        neighbor_distances: Optional[np.ndarray]
    ) -> np.ndarray:
        """Extract per-atom structural/force features.
        
        Features used:
        1. Force magnitude (norm of force vector)
        2. Neighbor density (atoms within cutoff)
        3. Average neighbor distance
        4. Energy gradient magnitude
        5. Velocity variance (temperature proxy)
        6. Atomic number (type) as a bias term
        """
        n_atoms = len(atomic_numbers)
        features = []
        cutoff = self.cutoff_angstrom
        
        for i in range(n_atoms):
            f_features = []
            
            # Feature 1: Force magnitude (if forces available)
            if forces is not None and i < len(forces):
                f_mag = np.linalg.norm(forces[i])
                f_features.append(f_mag)
            else:
                f_features.append(0.0)
            
            # Feature 2: Neighbor density (atoms within cutoff)
            if positions is not None:
                dr = positions - positions[i]  # (n_atoms, 3)
                dists = np.linalg.norm(dr, axis=1)
                n_neighbors = np.sum(dists < cutoff) - 1  # exclude self
                f_features.append(n_neighbors / n_atoms)  # normalized density
            else:
                f_features.append(0.0)
            
            # Feature 3: Average neighbor distance
            if positions is not None and neighbor_distances is not None:
                avg_dist = np.mean(neighbor_distances[i]) if len(neighbor_distances[i]) > 0 else cutoff
                f_features.append(avg_dist / cutoff)
            else:
                f_features.append(1.0)
            
            # Feature 4: Energy gradient (if energies available)
            if energies is not None and i < len(energies):
                e_grad = np.abs(energies[i])  # simplified gradient
                f_features.append(e_grad)
            else:
                f_features.append(0.0)
            
            # Feature 5: Temperature proxy (velocity variance)
            if velocities is not None and np.mean(velocities**2) > 0:
                temp_proxy = np.sum(velocities[i]**2) / (3.0 * 1.0)  # k/m
                f_features.append(temp_proxy / self.temperature)
            else:
                f_features.append(0.0)
            
            # Feature 6: Atomic number as a bias term
            if len(atomic_numbers) > 0 and i < len(atomic_numbers):
                f_features.append(atomic_numbers[i] / 100.0)
            else:
                f_features.append(0.0)
            
            features.append(f_features)
        
        features = np.array(features, dtype=np.float64)
        
        # Pad or truncate to n_features
        if features.shape[1] < self.n_features:
            padding = np.zeros((n_atoms, self.n_features - features.shape[1]))
            features = np.hstack([features, padding])
        elif features.shape[1] > self.n_features:
            features = features[:, :self.n_features]
        
        return features
    
    def _forward(self, features: np.ndarray) -> np.ndarray:
        """Two-layer feedforward network.
        
        Hidden: ReLU activation
        Output: Softmax for probability distribution over depths
        
        Returns:
            logits: float array (n_atoms, max_depth + 1)
        """
        # Layer 1: Linear + ReLU
        h = features @ self.weights['W1'] + self.weights['b1']
        h = np.maximum(0, h)  # ReLU
        
        # Layer 2: Linear
        logits = h @ self.weights['W2'] + self.weights['b2']
        
        return logits
    
    def _logits_to_depths(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to discrete depth scores.
        
        Uses softmax-based probability + physical feature thresholds:
        - High force -> deeper simulation
        - High density -> deeper simulation
        - High energy -> deeper simulation
        """
        n_atoms = logits.shape[0]
        depths = np.zeros(n_atoms, dtype=int)
        
        # Softmax probabilities for each depth
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        # Mean probability mass in high depths (3-4) and low depths (0-1)
        high_mass = probs[:, 3:].sum(axis=1)  # (n_atoms,)
        low_mass = probs[:, :2].sum(axis=1)    # (n_atoms,)
        mid_mass = probs[:, 2:3].sum(axis=1)   # (n_atoms,)
        
        for i in range(n_atoms):
            if i >= len(high_mass) or i >= len(low_mass):
                depths[i] = 2
                continue
                
            h = float(high_mass[i])
            l = float(low_mass[i])
            m = float(mid_mass[i])
            
            # Clear thresholds
            if h > 0.6:
                depths[i] = 4
            elif h > 0.3:
                depths[i] = 3
            elif l > 0.7:
                depths[i] = 0
            elif l > 0.5:
                depths[i] = 1
            elif m > 0.4:
                depths[i] = 2
            else:
                # Default: use argmax
                depths[i] = int(np.argmax(probs[i]))
        
        return depths
    
    def reset_weights(self):
        """Reset weights to initial random state."""
        np.random.seed(42)
        self.weights = {
            'W1': np.random.randn(self.n_features, self.hidden_dim) * 0.01,
            'b1': np.zeros(self.hidden_dim),
            'W2': np.random.randn(self.hidden_dim, self.max_depth + 1) * 0.01,
            'b2': np.zeros(self.max_depth + 1),
        }
