# Spoken Digit Recognition with Spiking Heidelberg Digits (SHD)

## Overview

This project implements a bio-inspired spoken digit recognition system using the [Spiking Heidelberg Digits (SHD) dataset](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/). Unlike traditional CNNs, this model uses a **Spiking Neural Network (SNN)** consisting of Leaky Integrate-and-Fire (LIF) neurons. The system processes asynchronous spike events from 700 input channels and learns via **Surrogate Gradient Descent**, achieving efficient temporal pattern recognition.

Key focuses:

* **Bio-Inspired Dynamics**: Implementation of LIF neurons with decay constants ( and ) for synaptic and membrane potentials.
* **Surrogate Gradients**: Overcoming the non-differentiable spiking threshold using a fast-sigmoid surrogate function.
* **Efficiency**: Sparse-to-dense data generation to handle high-dimensional event data without RAM exhaustion.
* **Performance**: Optimized tensor operations using `torch.einsum` for fast recurrent SNN simulation on GPUs.

## Dataset

* **Source**: Zenke Lab SHD (~10,000 alignment-free spike recordings).
* **Structure**: HDF5 files containing spike times and unit IDs (700 input channels). 20 classes (digits 0-9 in English and German).
* **Preprocessing**: Discretized into 100 temporal bins ( resolution), normalized, and converted into sparse tensors for memory-efficient batching.

## Features and Improvements

* **Preprocessing Pipeline**: Custom HDF5 loader that handles compressed spike data and converts it into sparse coordinate (COO) tensors.
* **Model**: Recurrent SNN (700-128-20) with recurrent hidden connections () to capture long-term temporal dependencies in audio.
* **Training**: Adam optimizer with Surrogate Gradient Backpropagation Through Time (BPTT).
* **Evaluation**: Classification accuracy based on the "Maximum Membrane Potential" (Voltage Peak) of the output layer over the simulation window.
* **Reproducibility**: Standardized weight initialization (Xavier-style for SNNs) and automated data path management for Colab/Kaggle environments.

## Requirements

* Python 3.8+ (tested on 3.12)
* Libraries:
* `torch`
* `h5py` (for dataset access)
* `numpy`
* `matplotlib`
* `utils.py` (Zenke Lab spiking dataset utilities)



Install via: `pip install torch h5py numpy matplotlib`

## Setup

### On Google Colab / Kaggle

1. **Prepare Data**: Run the provided `wget` and `gunzip` commands to download the SHD HDF5 files into the `/content/data` or `/kaggle/working` directory.
2. **Environment**: Ensure the `utils.py` file from the SpyTorch repository is in your working directory.
3. **Hardware**: Enable **GPU** (T4 or higher). The `run_snn` function is optimized for CUDA device tensors.
4. **Execution**: Run the initialization cells to define the `SurrGradSpike` class and weight tensors before calling the `train()` loop.

## Usage

### Training

* Run the `train()` function. It will iterate through the dataset using the `sparse_data_generator_from_hdf5_spikes`.
* Monitors cross-entropy loss and outputs epoch-wise performance.
* Artifacts: Trained weight tensors (`w1`, `w2`, `v1`) and `loss_hist`.

### Visualization

The project includes specialized plotting for spiking data:

* **Loss Curves**: Standard training convergence plots.
* **Raster Plots**: Visualization of hidden layer spiking activity over time.
* **Membrane Traces**: Monitoring the "leak and integrate" behavior of specific neurons.

```python
# Example: Visualizing a Spike Raster
plot_spike_raster(batch_idx=0)

```

## Project Structure

* **SurrGradSpike**: Custom `torch.autograd.Function` for the surrogate gradient.
* **sparse_data_generator**: Handles the lazy-loading of HDF5 spike data.
* **run_snn**: The temporal loop that simulates the LIF neuron equations across time steps.
* **compute_classification_accuracy**: Batched evaluation logic for the test set.

## Results

* **Expected Performance**: Stable convergence to >70% accuracy within 15 epochs (baseline for simple recurrent SNN on SHD).
* **Sparsity**: The network achieves recognition using significantly fewer "activations" than a standard ANN, demonstrating the power of event-based processing.
* **Runtime**: ~1 minute per epoch on a standard T4 GPU.

## Credits

* Dataset and Original Spytorch Tutorial: [Friedemann Zenke](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).
* Built using iterative refinements for modern NumPy compatibility and cloud-based absolute pathing.
