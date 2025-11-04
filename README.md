# Assignment 3 Instructions

This repository contains the implementation and analysis for Assignment 3. Additional documentation can be found in `final/doc/readme.txt`.

## Setup Instructions

### 1. Create Virtual Environment

**All OSes (Unix-like and Window):**

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

#### All OSes (Unix-like and Windows)

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Code

The main code is located in the `final/code/` directory:

- `algorithms/` - Contains all algorithm implementations
- `main/` - Contains the main execution script

### Configuration

Before running, configure your desired algorithm in `final/code/utilities/config.py`. Simply uncomment the algorithm(s) you want to test. The program will generate a `.zip` file containing IOH data in `final/doc/data/`.

### Execution

Once you've configured the settings, make sure to cd into `final/code/main` and run:

```bash
python3 main.py
```

<!-- ## Project Structure

### Algorithms

All algorithm implementations: `final/code/algorithms/`

### Documentation & Analysis

- **Plots & Analysis**: `final/doc/analysis/Assignment_2_Analysis.pdf`
- **Team Contribution**: `final/doc/team_contribution.txt`

### Data

Backup IOH data files: `final/doc/data/` -->
