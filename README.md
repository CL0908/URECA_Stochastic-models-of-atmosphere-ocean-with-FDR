# Stochastic-models-of-atmosphere-ocean-with-FDR

URECA Research Project
Stochastic Atmosphere–Ocean Interaction Models
<div align="center"> <h3>Fluctuation–Dissipation Relations in Climate Dynamics</h3> <p> Simulation code for stochastic atmosphere–ocean models developed under the <b>URECA project SPMS25003</b> </p> <p> Spectral stability • Lyapunov equations • Stationary distributions </p> </div>

# About the Project
This repository contains simulation and analysis code developed for the URECA research project
SPMS25003 — Simplified Models of Atmosphere–Ocean Interaction with Fluctuation and Dissipation
Supervised by
**Assoc Prof François Joachim Marcel Gay-Balmaz**
School of Physical and Mathematical Sciences
**Nanyang Technological University**
The project investigates the mathematical structure of fluctuation–dissipation relations (FDR) in stochastic models of atmosphere–ocean interaction.
We study simplified two-layer stochastic climate models where atmospheric and oceanic velocities interact through coupling forces and stochastic forcing.
The goal is to understand how
spectral stability
dissipation mechanisms
stochastic forcing
determine the existence and structure of stationary invariant measures.

# Approach
The simulations analyze several stochastic dynamical systems including:
Linear two-layer stochastic model
Study of the drift operator spectrum and the role of damping in restoring stability.
Ornstein–Uhlenbeck shear dynamics
Closed-form dynamics for the shear mode with explicit variance predictions.
Lyapunov equation analysis
The stationary covariance matrix is obtained from the continuous Lyapunov equation
A Σ + Σ Aᵀ + Q = 0
connecting stochastic forcing with dissipative structure.
Quadratic drag extension
A nonlinear drag model relevant to geophysical flows is studied and compared with an effective linear eddy-friction approximation.

# Results
Numerical simulations confirm several theoretical predictions:
undamped systems possess neutral modes preventing stationary measures
introducing damping restores Hurwitz stability
the stationary covariance predicted by the Lyapunov equation matches simulation
shear velocity follows a Boltzmann-type stationary distribution
Simulation results verify the fluctuation–dissipation balance between stochastic forcing and dissipative dynamics.

# Repository Contents
.
├── simulation
│   ├── linear_model.py
│   ├── quadratic_drag.py
│   └── parameter_estimation.py
│
├── analysis
│   ├── covariance_diagnostics.py
│   └── lyapunov_solver.py
│
├── figures
│   ├── graph1.png
│   └── graph2.png
│   └── ....png
│
└── README.md

# Project Information
Project
SPMS25003 – Simplified Models of Atmosphere–Ocean Interaction with Fluctuation and Dissipation
Programme
URECA (Undergraduate Research Experience on Campus)
Category of Participation
AU
Supervisor
Assoc Prof François Joachim Marcel Gay-Balmaz
Institution
Nanyang Technological University
