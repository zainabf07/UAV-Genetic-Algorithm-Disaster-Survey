# UAV Disaster-Surveillance Strategy Engine Using a Genetic Algorithm
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![AI](https://img.shields.io/badge/Domain-Artificial%20Intelligence-purple)
![Algorithm](https://img.shields.io/badge/Algorithm-Genetic%20Algorithm-green)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange)
![Status](https://img.shields.io/badge/Status-Academic%20Project-lightgrey)
![License](https://img.shields.io/badge/License-Educational-informational)

**Course:** Artificial Intelligence — BSCS 5  
**Assignment:** Assignment 2  
**Department:** Computer Science  
**Institution:** Bahria University, Lahore Campus  
**Project Type:** AI Optimization and Simulation

## 1. Project Overview

This project implements a Genetic Algorithm (GA) to optimize the deployment and route planning of three Unmanned Aerial Vehicles (UAVs) in a simulated disaster-surveillance environment.

The environment is represented as a **50 × 50 grid** containing open cells, restricted areas, environmental hazards, and high-priority disaster zones. Each UAV follows a route defined by six waypoints, with configurable speed and sensor-power settings.

The objective is to identify candidate plans that balance area coverage, target detection, energy consumption, flight time, and environmental risk.

Because these objectives may conflict, the system uses a weighted fitness function to evaluate and improve candidate solutions over successive generations.

> **Note:** This project is an educational simulation. Its route, energy, hazard, and sensor models are simplified and are not intended for real-world UAV operations.

## 2. Project Objectives

The main objectives are to:

- Design a chromosome representation for multiple UAVs.
- Implement a multi-objective fitness function.
- Apply selection, crossover, mutation, and elitism.
- Handle coordinate bounds, no-fly zones, and flight-range constraints.
- Visualize fitness convergence and evolved flight paths.
- Analyze the effects of GA hyperparameters.
- Compare the GA with a random-search baseline.

## 3. Key Features

- 50 × 50 simulated disaster grid.
- Three UAVs with fixed starting positions.
- Six waypoints per UAV.
- Real-valued chromosome encoding.
- Priority-weighted coverage and a detection proxy.
- Energy, flight-time, and hazard-cost estimates.
- Tournament selection and elitism.
- Segment-based crossover and Gaussian mutation.
- Waypoint repair and constraint penalties.
- Fitness convergence graph.
- UAV route visualization.
- Sensor coverage heatmap.
- JSON files for metrics and experiment history.

## 4. Genetic Algorithm Design

### 4.1 Chromosome Representation

Each chromosome has the shape:

`(3, 6, 4)`

This represents three UAVs, six waypoints per UAV, and four values per waypoint.

| Gene | Description | Allowed range |
|---|---|---|
| `x` | Horizontal grid coordinate | 0–49 |
| `y` | Vertical grid coordinate | 0–49 |
| `speed` | Normalized speed setting | 0.5–1.5 |
| `sensor_power` | Sensor-power setting | 0–1 |

Therefore, each chromosome contains 72 numerical values.

### 4.2 Fitness Function

The system maximizes the following weighted objective:

\[
F=0.48C+0.22D-0.13E-0.07T-0.10R-0.02V
\]

Where:

- \(C\): Priority-weighted area coverage.
- \(D\): Simplified target-detection score.
- \(E\): Normalized energy-cost estimate.
- \(T\): Normalized flight-time estimate.
- \(R\): Mean environmental hazard exposure.
- \(V\): Constraint-violation score.

Coverage and detection contribute positively, while energy, time, hazard exposure, and constraint violations reduce fitness.

A higher fitness score is preferred. The weights are design choices and may be adjusted according to mission priorities.

### 4.3 Genetic Operators

The implementation uses:

1. **Population initialization:** Generates random candidate plans.
2. **Tournament selection:** Selects parents based on fitness.
3. **Crossover:** Combines waypoint segments from parent chromosomes.
4. **Mutation:** Perturbs coordinates, speed, and sensor-power settings.
5. **Elitism:** Preserves the best candidates across generations.
6. **Termination:** Stops after the configured number of generations and returns the best solution found.

### 4.4 Constraint Handling

The implementation clips coordinates and control values to their permitted ranges and attempts to relocate waypoints that fall inside blocked cells.

During evaluation, sampled route segments are checked for no-fly-zone incursions, and penalties are applied for incursions and excessive total route distance.

Waypoint repair alone does not guarantee that every route segment is obstacle-free.

## 5. Technologies Used

- **Python:** Core implementation.
- **NumPy:** Numerical computation and array operations.
- **Matplotlib:** Fitness graphs and route visualizations.
- **JSON:** Storage of metrics and experiment history.
- **python-docx:** Generation of the technical report.

## 6. System Requirements

- Python 3.10 or newer recommended.
- NumPy.
- Matplotlib.
- python-docx.

## 7. Installation and Execution

### Step 1: Download or Clone the Repository

Download the project ZIP from the assignment files and extract it, or clone the repository if it has been uploaded to GitHub.

### Step 2: Open the Project

Open the extracted project directory in Visual Studio Code or another Python-compatible editor.

### Step 3: Install Dependencies

Open a terminal in the project root and run:

```bash
python -m pip install -r requirements.txt
```

On Windows, you can use `py` instead of `python` if necessary.

### Step 4: Run the Simulation

Execute:

```bash
python src/uav_ga.py
```

The program runs the Genetic Algorithm, prints the best solution's metrics, and generates output files in the `results/` directory.

### Step 5: Review the Results

Inspect the generated graphs, JSON metrics, and best chromosome to understand the quality of the evolved solution.

## 8. Project Structure

```text
UAV_GA_Assignment2/
├── README.md
├── requirements.txt
├── src/
│   └── uav_ga.py
├── results/
│   ├── best_chromosome.npy
│   ├── metrics.json
│   ├── history.json
│   ├── fitness_convergence.png
│   ├── uav_routes.png
│   └── coverage_heatmap.png
└── docs/
    ├── technical_report.docx
    ├── fitness_convergence.png
    ├── uav_routes.png
    └── coverage_heatmap.png
```

The contents of `results/` depend on which experiments have been executed. Some analysis files may be generated separately from the main simulation.

## 9. Output Visualizations

### Fitness Convergence

`fitness_convergence.png` displays the best and average fitness across generations. It helps assess whether the population is improving over time.

### UAV Flight Paths

`uav_routes.png` displays the evolved routes of the three UAVs over the simulated priority zones and restricted regions.

### Sensor Coverage Heatmap

`coverage_heatmap.png` illustrates the distribution of simulated sensor-footprint visits across the grid.

These visualizations should be interpreted together with the numerical metrics. An improvement in total fitness does not necessarily mean every individual objective has improved.

## 10. Hyperparameter Sensitivity Analysis

The algorithm's performance depends on several configuration parameters:

- **Population size:** Controls the number of candidate solutions explored per generation.
- **Generations:** Controls the number of evolutionary iterations.
- **Mutation rate:** Influences how frequently genes are perturbed.
- **Tournament size:** Controls selection pressure.
- **Random seed:** Helps make experiments reproducible.

For sensitivity analysis, change one parameter at a time while keeping the others fixed. Record fitness, coverage, energy, hazard exposure, and runtime.

Since the GA is stochastic, repeat each configuration using multiple random seeds and report the mean and standard deviation for stronger experimental evidence.

## 11. Comparison with Random Search

Random search generates candidate solutions independently and retains the best candidate evaluated. The Genetic Algorithm additionally uses selection, crossover, mutation, and elitism to exploit promising solutions while continuing to explore alternatives.

For a fair comparison:

- Use the same fitness function and environment.
- Use comparable candidate-evaluation budgets.
- Repeat both approaches over multiple random seeds.
- Report the best, mean, and variability of fitness scores.

A single experiment is illustrative and does not establish that one approach is always superior.

## 12. Assumptions and Limitations

This project is designed to demonstrate Genetic Algorithm concepts rather than provide a complete operational UAV system.

Important limitations include:

- Synthetic disaster maps and hazard fields.
- Simplified sensor-detection and energy models.
- Approximate flight-time calculations.
- Straight-line route segments sampled at finite intervals.
- No complete model of wind, battery discharge, communications, or inter-UAV collision avoidance.
- No guarantee of globally optimal solutions or fully safe real-world paths.

Real deployment would require validated maps, realistic vehicle and sensor models, robust obstacle avoidance, coordination mechanisms, and independent safety verification.

## 13. Conclusion

This project demonstrates how Genetic Algorithms can be applied to a multi-objective UAV route-planning problem. It integrates chromosome design, fitness evaluation, genetic operators, constraint handling, visualization, and experimental analysis into a modular Python implementation.

The project provides a foundation for further work in evolutionary optimization, autonomous navigation, and intelligent disaster-surveillance planning.

---
