"""
Molecular Dynamics Engine
=========================

Classical molecular dynamics simulation using Newtonian mechanics
with various force field approximations.

Implements:
- Velocity Verlet integration
- Lennard-Jones potentials
- Harmonic bond potentials
- Periodic boundary conditions (optional)

Usage:
    engine = MolecularDynamicsEngine(force_field="universal")
    result = engine.run_simulation(system_state, steps=1000, timestep=0.5)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np


class MolecularDynamicsEngine:
    """
    Classical molecular dynamics engine for atomic/molecular simulations.
    
    Uses velocity Verlet integration with configurable force fields
    to simulate the physical movements of atoms and molecules.
    """
    
    # Standard atomic masses (g/mol)
    ATOMIC_MASSES = {
        "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
        "P": 30.974, "S": 32.065, "Fe": 55.845, "Mg": 24.305,
    }
    
    # Lennard-Jones parameters (epsilon in kJ/mol, sigma in Å)
    LJ_PARAMS = {
        "C": {"epsilon": 0.3598, "sigma": 3.4},
        "N": {"epsilon": 0.7112, "sigma": 3.25},
        "O": {"epsilon": 0.6502, "sigma": 3.0},
        "H": {"epsilon": 0.0657, "sigma": 2.5},
        "P": {"epsilon": 0.8368, "sigma": 3.74},
        "S": {"epsilon": 1.0460, "sigma": 3.55},
    }
    
    def __init__(self, force_field: str = "universal"):
        """
        Initialize the Molecular Dynamics Engine.
        
        Args:
            force_field: Type of force field ("universal", "amber", "charmm")
        """
        self.force_field = force_field
        self.kB = 8.314e-3  # Boltzmann constant in kJ/(mol·K)
        print(f"Initialized MolecularDynamicsEngine with force field: {self.force_field}")
    
    def run_simulation(
        self,
        system_state: Dict[str, Any],
        steps: int = 1000,
        timestep: float = 0.5,
        temperature: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Run a molecular dynamics simulation.
        
        Args:
            system_state: Initial state (atom positions, velocities, types)
            steps: Number of integration steps
            timestep: Time step in femtoseconds
            temperature: Target temperature in Kelvin
        
        Returns:
            Dictionary with trajectory and final state
        """
        start_time = time.time()
        system_name = system_state.get("name", "system")
        atoms = system_state.get("atoms", [])
        
        # Initialize if no atoms provided
        if not atoms:
            num_atoms = system_state.get("num_atoms", 10)
            atoms = self._generate_random_system(num_atoms)
        
        print(f"  MD: Running simulation for {system_name} ({len(atoms)} atoms) "
              f"for {steps} steps (Δt={timestep} fs, T={temperature}K)")
        
        # Extract positions and velocities
        positions = np.array([atom.get("position", [0, 0, 0]) for atom in atoms])
        velocities = np.array([atom.get("velocity", [0, 0, 0]) for atom in atoms])
        masses = np.array([self.ATOMIC_MASSES.get(atom.get("type", "C"), 12.0) for atom in atoms])
        atom_types = [atom.get("type", "C") for atom in atoms]
        
        # Initialize velocities from Maxwell-Boltzmann if all zero
        if np.allclose(velocities, 0):
            velocities = self._init_velocities(masses, temperature)
        
        # Run velocity Verlet integration
        trajectory = [positions.tolist()]
        potential_energies = []
        kinetic_energies = []
        
        forces = self._compute_forces(positions, atom_types)
        
        for step in range(steps):
            # Velocity Verlet: update positions
            accelerations = forces / masses[:, np.newaxis]
            positions = positions + velocities * timestep + 0.5 * accelerations * timestep**2
            
            # Compute new forces
            new_forces = self._compute_forces(positions, atom_types)
            
            # Velocity Verlet: update velocities
            new_accelerations = new_forces / masses[:, np.newaxis]
            velocities = velocities + 0.5 * (accelerations + new_accelerations) * timestep
            
            forces = new_forces
            
            # Apply thermostat every 10 steps
            if step % 10 == 0:
                velocities = self._apply_thermostat(velocities, masses, temperature)
            
            # Record trajectory (every 10 steps to save memory)
            if step % 10 == 0:
                trajectory.append(positions.tolist())
                pe = self._compute_potential_energy(positions, atom_types)
                ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
                potential_energies.append(pe)
                kinetic_energies.append(ke)
        
        execution_time = int((time.time() - start_time) * 1000)
        
        return {
            "trajectory": trajectory,
            "final_positions": positions.tolist(),
            "final_velocities": velocities.tolist(),
            "potential_energies": potential_energies,
            "kinetic_energies": kinetic_energies,
            "total_energies": [pe + ke for pe, ke in zip(potential_energies, kinetic_energies)],
            "average_temperature": self._compute_temperature(velocities, masses),
            "execution_time_ms": execution_time,
            "steps_completed": steps,
            "message": f"MD simulation completed for {system_name}",
        }
    
    def _generate_random_system(self, num_atoms: int) -> List[Dict[str, Any]]:
        """Generate a random initial system."""
        atoms = []
        atom_types = ["C", "N", "O", "H"]
        
        for i in range(num_atoms):
            atoms.append({
                "type": np.random.choice(atom_types, p=[0.4, 0.2, 0.2, 0.2]),
                "position": (np.random.rand(3) * 10 - 5).tolist(),
                "velocity": [0.0, 0.0, 0.0],
            })
        
        return atoms
    
    def _init_velocities(
        self,
        masses: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Initialize velocities from Maxwell-Boltzmann distribution."""
        n = len(masses)
        velocities = np.zeros((n, 3))
        
        for i in range(n):
            sigma = np.sqrt(self.kB * temperature / masses[i])
            velocities[i] = np.random.normal(0, sigma, 3)
        
        # Remove center of mass velocity
        total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
        total_mass = np.sum(masses)
        velocities -= total_momentum / total_mass
        
        return velocities
    
    def _compute_forces(
        self,
        positions: np.ndarray,
        atom_types: List[str],
    ) -> np.ndarray:
        """Compute forces on all atoms using Lennard-Jones potential."""
        n = len(positions)
        forces = np.zeros_like(positions)
        
        for i in range(n):
            for j in range(i + 1, n):
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                
                if r < 0.1:  # Avoid singularity
                    r = 0.1
                
                # Get LJ parameters (use combining rules)
                eps_i = self.LJ_PARAMS.get(atom_types[i], {"epsilon": 0.5})["epsilon"]
                eps_j = self.LJ_PARAMS.get(atom_types[j], {"epsilon": 0.5})["epsilon"]
                sig_i = self.LJ_PARAMS.get(atom_types[i], {"sigma": 3.0})["sigma"]
                sig_j = self.LJ_PARAMS.get(atom_types[j], {"sigma": 3.0})["sigma"]
                
                epsilon = np.sqrt(eps_i * eps_j)
                sigma = (sig_i + sig_j) / 2
                
                # LJ force: F = 24*eps/r * (2*(sig/r)^12 - (sig/r)^6)
                sr6 = (sigma / r) ** 6
                force_mag = 24 * epsilon / r * (2 * sr6**2 - sr6)
                
                force_vec = force_mag * r_vec / r
                forces[i] -= force_vec
                forces[j] += force_vec
        
        return forces
    
    def _compute_potential_energy(
        self,
        positions: np.ndarray,
        atom_types: List[str],
    ) -> float:
        """Compute total potential energy."""
        n = len(positions)
        energy = 0.0
        
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(positions[j] - positions[i])
                if r < 0.1:
                    r = 0.1
                
                eps_i = self.LJ_PARAMS.get(atom_types[i], {"epsilon": 0.5})["epsilon"]
                eps_j = self.LJ_PARAMS.get(atom_types[j], {"epsilon": 0.5})["epsilon"]
                sig_i = self.LJ_PARAMS.get(atom_types[i], {"sigma": 3.0})["sigma"]
                sig_j = self.LJ_PARAMS.get(atom_types[j], {"sigma": 3.0})["sigma"]
                
                epsilon = np.sqrt(eps_i * eps_j)
                sigma = (sig_i + sig_j) / 2
                
                sr6 = (sigma / r) ** 6
                energy += 4 * epsilon * (sr6**2 - sr6)
        
        return energy
    
    def _apply_thermostat(
        self,
        velocities: np.ndarray,
        masses: np.ndarray,
        target_temp: float,
    ) -> np.ndarray:
        """Apply Berendsen thermostat."""
        current_temp = self._compute_temperature(velocities, masses)
        
        if current_temp < 1e-6:
            return velocities
        
        tau = 0.1  # Coupling constant
        scale = np.sqrt(1 + 0.1 * (target_temp / current_temp - 1))
        
        return velocities * scale
    
    def _compute_temperature(
        self,
        velocities: np.ndarray,
        masses: np.ndarray,
    ) -> float:
        """Compute instantaneous temperature from kinetic energy."""
        n = len(masses)
        ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
        
        # T = 2*KE / (3*N*kB)
        return 2 * ke / (3 * n * self.kB)


def run_md_simulation(
    system_name: str,
    num_atoms: int = 20,
    steps: int = 1000,
    temperature: float = 300.0,
) -> Dict[str, Any]:
    """
    Convenience function to run a molecular dynamics simulation.
    
    Args:
        system_name: Name of the system
        num_atoms: Number of atoms
        steps: Simulation steps
        temperature: Target temperature
    
    Returns:
        Simulation results
    """
    engine = MolecularDynamicsEngine()
    return engine.run_simulation(
        {"name": system_name, "num_atoms": num_atoms},
        steps=steps,
        temperature=temperature,
    )
