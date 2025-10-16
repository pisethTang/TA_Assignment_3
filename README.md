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

## Experimenting

Before we move on any further, some comments -- this is a suggestion (and a good practice as well) that you guys should give a try before executing the main script. I have made another notebook file named `exercise3_sketches.ipynb` in `/ideas/`. It should give you a decent starting point at having a better control on the output of your assigned to-be-implemented algorithms. It imports the settings from `final/code/utilities/config.py` so after you implement your algorithms inside `final/code/algorithms/`, and import them into `config.py`, just come to the notebook to run it on the problem instances. Once you are satisfied with the output and plots from the IOHAnalyzer, you can move on to the next step. But you don't have to follow this strictly if you don't want to -- you can execute the main script right away. Note that there could be linting errors inside `exercse3_sketches.ipynb`, we just have to select the right environment for our formmatter to detect -- `.venv (Python 3.10.12)` or the one recommended by your system. Moreover, there could also be annoying linting errors in other scripts, just select the python interpreter recommended by your system and they should go away. And you may need to install `ipykernel` recommended by the notebook itself.

## Running the Code

The main code is located in the `final/code/` directory:

- `algorithms/` - Contains all algorithm implementations
- `main/` - Contains the main execution script

### Configuration

Before running, configure your desired algorithm in `final/code/utilities/config.py`. Simply uncomment the algorithm you want to test. The program will generate a `.zip` file containing IOH data in `final/doc/data/`.

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
