# CS245 Track A: User Simulation Agent

**Enhanced User Simulation with Multi-Component Architecture**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Ablation Study Methodology](#ablation-study-methodology)
- [Evaluation Results](#evaluation-results)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project implements an enhanced user simulation agent for Track A of the CS245 course. The agent generates personalized reviews and ratings by modeling user behavior patterns, considering review quality factors, and employing score calibration techniques.

### Key Results

| Dataset | RMSE | MAE | Pearson | Overall Quality |
|---------|------|-----|---------|-----------------|
| **Yelp** | 0.968 | 0.745 | 0.691 | 82.8% |
| **Amazon** | 1.024 | 0.798 | 0.658 | 80.3% |
| **Goodreads** | 0.892 | 0.681 | 0.723 | 85.0% |

---

## 🏗️ Architecture

Our agent employs a **multi-component architecture** with four core modules:

```
┌─────────────────────────────────────────────────────────┐
│                  Enhanced Agent                         │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   User       │  │   Quality    │  │  Reflection  │  │
│  │  Profiling   │  │   Analysis   │  │   Mechanism  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Score Calibration Module                 │  │
│  └──────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│              Base LLM (DeepSeek Chat)                   │
└─────────────────────────────────────────────────────────┘
```

### Component Descriptions

1. **User Profiling Module**
   - Analyzes historical review patterns
   - Identifies user rating tendencies (lenient/critical/neutral)
   - Tracks engagement style (useful/funny/cool metrics)
   - Maintains cross-task user history

2. **Quality Analysis Module**
   - Evaluates reference reviews for informativeness
   - Identifies high-quality review examples
   - Incorporates useful/funny/cool features
   - Guides content generation style

3. **Reflection Mechanism**
   - Validates sentiment-rating consistency
   - Iteratively refines generated content
   - Ensures logical coherence

4. **Score Calibration Module**
   - Corrects LLM prediction bias
   - Aligns predictions with actual rating distributions
   - Reduces systematic over/under-estimation

---

## ✨ Features

- ✅ **Personalized User Modeling**: Deep analysis of individual user characteristics
- ✅ **Quality-Aware Generation**: Considers community engagement metrics
- ✅ **Cross-Task Memory**: Accumulates knowledge across simulation tasks
- ✅ **Score Calibration**: Reduces prediction deviation from ground truth
- ✅ **Multi-Dataset Support**: Works with Yelp, Amazon, and Goodreads
- ✅ **Comprehensive Evaluation**: Full metrics suite with quality assessment

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- DeepSeek API key ([Get one here](https://platform.deepseek.com))

### Step 1: Clone Repository

```bash
git clone <your-repository-url>
cd cs245-track-a
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

```bash
# Download and extract the CS245 Track A dataset
# Place in ./Dataset/ directory
# Expected structure:
# Dataset/
#    item.json
#    review.json
#    user.json
# !!!!! Note that the dataset must contain all data from the three datasets to utilize the full functionality.!!!!!
```

### Step 5: Configure API Key

```bash
# Option 1: Set environment variable
export DEEPSEEK_API_KEY="your-api-key-here"

# Option 2: Edit run_evaluation.py
# Set API_KEY = "your-api-key-here" in the configuration section
```

---

## 🚀 Quick Start

### Run Full Evaluation

```bash
# Evaluate on Yelp dataset (300 samples)
python run_evaluation.py

# The script will:
# 1. Initialize the simulator
# 2. Load tasks and ground truth
# 3. Run the enhanced agent
# 4. Calculate metrics
# 5. Save results to results/evaluation_results_yelp_TIMESTAMP.json
```

### Expected Output

```
================================================================================
STARTING EVALUATION
================================================================================
Dataset: yelp
Number of samples: 300

[Step 1/5] Initializing Simulator...
✓ Simulator initialized successfully

[Step 2/5] Loading tasks and groundtruth...
✓ Loaded 400 tasks
✓ Will evaluate 300 samples

[Step 3/5] Setting up agent...
✓ Agent class set: ImprovedSimulationAgent

[Step 4/5] Configuring LLM client...
✓ LLM client configured: DeepSeek Chat

[Step 5/5] Running simulation...
This will take approximately 2.5 minutes
--------------------------------------------------------------------------------
INFO: Execution plan generated: 3 steps
INFO: User analysis complete: critical, engagement_style: high
INFO: Quality analysis: useful_examples=True, funny_examples=False
INFO: Starting review generation (reflection mode: True)
INFO: Review generation complete: 4.0 stars, length 287 characters
[Progress: 1/300 completed]
...
--------------------------------------------------------------------------------
✓ Simulation completed in 162.34 seconds
✓ Average time per task: 0.54s

[Saving Results]
✓ Results saved to: results/evaluation_results_yelp_20251127_183045.json

================================================================================
EVALUATION SUMMARY
================================================================================
RMSE: 0.9680
MAE: 0.7450
Pearson Correlation: 0.6910
Accuracy (±1.0): 0.9030
Overall Quality: 0.8280
================================================================================

✓ Evaluation completed successfully!
```

### Inspect Individual Outputs

```bash
# View agent output for a specific task
python inspect_agent_output.py

# Interactive prompts:
# Enter task index (0-399): 42
# 
# Output:
# ⭐ Predicted Rating: 4.0
# 🎯 Actual Rating: 4.0
# 📊 Error: 0.00
# 
# 📝 Generated Review:
# ----------------------------------------------------------------------
# This place exceeded my expectations! The food was delicious and the
# service was friendly. I especially loved the atmosphere - cozy and
# inviting. Will definitely come back!
# ----------------------------------------------------------------------
```

---

## 🔬 Ablation Study Methodology

### Overview

To validate the contribution of each architectural component, we conducted **systematic ablation studies** by **selectively disabling individual modules** in the agent's reasoning pipeline.

#### Workflow with Conditional Execution

Each component is conditionally executed in the `workflow()` method based on its enable flag:

```python
def workflow(self):
    """Main workflow with conditional component execution"""
    
    # ... Load task data ...
    
    # COMPONENT 1: User Profiling (Conditional)
    if self.enable_user_profiling:
        # Analyze user's historical patterns
        user_profile = self.profile_analyzer.analyze_user_patterns(
            user_reviews
        )
        user_profile_text = self.profile_analyzer.format_user_analysis(
            user_profile
        )
    else:
        # Use generic default profile
        user_profile = self._get_default_profile()
        user_profile_text = "Generic user profile"
    
    # COMPONENT 2: Quality Analysis (Conditional)
    if self.enable_quality_analysis:
        # Analyze review quality features
        quality_analysis = self.quality_analyzer.analyze_review_qualities(
            reference_reviews
        )
    else:
        # Skip quality-aware features
        quality_analysis = None
    
    # Build prompt with available components
    prompt = self.build_prompt(
        user_profile=user_profile_text,
        quality_analysis=quality_analysis,
        # ... other context ...
    )
    
    # Generate initial review
    result = self.reasoning(prompt)
    
    # COMPONENT 3: Reflection (Conditional)
    if self.enable_reflection:
        # Iteratively refine the output
        result = self.reflect_and_refine(result)
    # else: skip refinement, use initial output directly
    
    # Parse LLM output
    stars, review_text = self.parse_review_result(result)
    
    # COMPONENT 4: Score Calibration (Conditional)
    if self.enable_calibration:
        # Calibrate predicted score
        stars = self._calibrate_score(
            raw_stars=stars,
            review_text=review_text,
            user_profile=user_profile
        )
    # else: use raw LLM prediction without calibration
    
    return {"stars": stars, "review": review_text}
```

### Ablation Configurations

We evaluate six configurations to isolate each component's contribution:

| Configuration | User<br/>Profiling | Quality<br/>Analysis | Reflection | Score<br/>Calibration | Description |
|---------------|:------------------:|:--------------------:|:----------:|:---------------------:|-------------|
| **full_model** | ✓ | ✓ | ✓ | ✓ | All modules enabled (baseline) |
| **no_calibration** | ✓ | ✓ | ✓ | ✗ | Tests calibration impact |
| **no_user_profiling** | ✗ | ✓ | ✓ | ✓ | Tests user modeling impact |
| **no_reflection** | ✓ | ✓ | ✗ | ✓ | Tests refinement impact |
| **no_quality_analysis** | ✓ | ✗ | ✓ | ✓ | Tests quality features impact |
---


### Metrics Explanation

- **RMSE**: Root Mean Squared Error (lower is better, penalizes large errors)
- **MAE**: Mean Absolute Error (lower is better, average prediction error)
- **Pearson Correlation**: Linear correlation coefficient (higher is better, -1 to 1)
- **Accuracy ±1.0**: Percentage of predictions within 1 star of ground truth
- **Overall Quality**: Combined quality score (0-1 scale, holistic assessment)

---

## 📁 Project Structure

```
cs245-track-a/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── improved_agent_with_quality.py         # Main agent implementation
├── run_evaluation.py                      # Evaluation script
├── inspect_agent_output.py                # Output inspection tool
│
├── Dataset/                               # Dataset directory
│   item.json
│   review.json
│   user.json
│
├── example/                               # Task and groundtruth data
│   └── track1/
│       ├── yelp/
│       │   ├── tasks/
│       │   └── groundtruth/
│       ├── amazon/
│       └── goodreads/
│
```

---

## 💡 Usage Examples

### Example 1: Evaluate on Different Dataset

```python
# Edit run_evaluation.py configuration in file
    DATA_DIR = "Dataset"  
    TASK_SET = "yelp"     # yelp / amazon / goodreads
    API_KEY = "sk-abab919cdfae44deac4d21cb974aa4e0"  # 👈  API Key
    
    #
    NUM_TASKS = 30 
    
    
    ENABLE_THREADING = True  
    MAX_WORKERS = 10        
    
    
    USE_CACHE = True  
```
```bash
# Run evaluation
python run_evaluation.py
```


### Example 2: Analyze Specific Task

```bash
# Use the inspection tool
python inspect_agent_output.py

# When prompted, enter task index: 150
# The tool will:
# 1. Load the task
# 2. Run the agent
# 3. Display predicted vs actual rating
# 4. Show generated review
# 5. Calculate error metrics
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. API Key Error

```
Error: Chat API error: 401 Unauthorized
```

**Solution**: Verify your DeepSeek API key is correct and active.

```bash
export DEEPSEEK_API_KEY="sk-your-actual-key-here"
```

#### 2. Module Import Error

```
ModuleNotFoundError: No module named 'websocietysimulator'
```

**Solution**: Install the WebSocietySimulator package.

```bash
pip install websocietysimulator
# or
pip install -r requirements.txt
```

#### 3. Dataset Not Found

```
FileNotFoundError: Dataset/
```

**Solution**: Ensure dataset is downloaded and placed in correct directory.

```bash
# Expected structure:
Dataset/
    users.json
    items.json
    reviews.json
```

#### 4. Memory Error

```
MemoryError: Unable to allocate array
```

**Solution**: Reduce batch size or enable caching.

```python
simulator = Simulator(
    data_dir="Dataset",
    cache=True,  # Enable caching
    device="cpu"  # Use CPU if GPU memory insufficient
)
```

### Getting Help
- **Inspect intermediate results**: Use `inspect_agent_output.py` to examine individual outputs

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Last Updated**: November 27, 2024