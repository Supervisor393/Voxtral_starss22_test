# Lost in Time: Systematic Temporal Bias in Large Audio Language Models

Large Audio Language Models (LALMs) are widely used for audio understanding and multimodal reasoning, but their ability to predict event timings remains underexplored. This study examines temporal bias in LALMs, revealing a consistent misalignment in timestamp predictions. For example, when asked “At which second does the lecturer introduce the key formula?”, models often predict timestamps that are consistently earlier or later than the ground truth.  We find that temporal bias is common across models and datasets, increases with audio length, and varies by event type and position. We introduce the Temporal Bias Index (TBI) to measure and visualize this bias, highlighting the need for more temporally accurate LALM architectures.

## Table of Contents

- [Project Overview](# Project Overview)
- [Installation Requirements](# Installation Requirements)
- [Project Structure](# project-structure)
- [Usage](# usage)
- [Experimental Results](# experimental-results)
- [Contributors](# contributors)
- [License](# license)

## Project Overview

This repository contains the code for four key experiments from our paper on temporal bias in Large Audio Language Models. 

1. **Experiment 1**: The impact of audio length on temporal bias.
2. **Experiment 2**: The impact of event position on temporal bias.
3. **Experiment 3**: The impact of event type and duration time on temporal bias.
4. **Supplementary experiments 4**: Nonsense and Event Detection Capabilities of LALMs.

## Installation Requirements

### Software Requirements

- Python 3.7 or higher
- PyTorch 1.8 or higher
- NumPy 1.19 or higher
- Other dependencies can be installed via `requirements.txt`.

### Installation Steps

To get started, clone this repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Install required Python dependencies
pip install -r requirements.txt
```
