# CircuitNet

This repository contains an opensource implementation of the ICML 2023 Spotlight paper: CircuitNet: A Generic Neural Network to Realize Universal Circuit Motif Modeling". CircuitNet models universal circuit motifs through a combination of product and linear transformations, enabling it to capture complex neural interactions like feed-forward, mutual, feed-back, and lateral inhibition patterns.

## Overview

CircuitNet introduces Circuit Motif Units (CMUs) as its basic building blocks. Each CMU contains densely connected neurons that can model various biological circuit motifs through:
- Linear transformations for feed-forward and mutual motifs
- Product transformations for feed-back and lateral motifs
- ClipSin activation function optimized for stability

## Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/CircuitNet.git
cd CircuitNet

# Create and activate conda environment
conda create -n circuitnet python=3.8
conda activate circuitnet

# Install requirements
pip install -r requirements.txt
