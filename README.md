# SingleShot_Pty

A ptychography simulation and processing package (`ssPty`), with GPU
acceleration via CuPy.

## Features

- **RPI simulations** — set up and generate reflection zone plate (RZP)
  simulations for visible light and X-rays (`ssPty.RPI_sims`).
- **Wave propagation** — Fresnel propagation and Fourier padding/cropping,
  with CuPy and NumPy backends (`ssPty.wave_propagation_cupy`).
- **Fourier Shell Correlation** — resolution estimation utilities
  (`ssPty.fourier_shell_corr`).
- **Misc utilities** — FFT wrappers, image processing, complex interpolation,
  and plotting helpers (`ssPty.misc`).

## Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA 12.x (for the CuPy-accelerated code paths)

## Installation

```bash
# create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# install dependencies and the package in editable mode
pip install -r requirements.txt
```

This installs `ssPty` in editable mode (`-e .`) so local changes take effect
immediately.

## Usage

`RPI_simu.py` is an example script demonstrating the full simulation and
reconstruction workflow. Import the package modules directly, for example:

```python
from ssPty.RPI_sims.setup_rzp_sims_vis import create_initial_object
from ssPty.fourier_shell_corr.fourier_shell_corr import fourier_shell_corr
from ssPty.wave_propagation_cupy.fourier_padding_cupy import fourier_pad, fourier_crop
```

## Package layout

```
ssPty/
├── engines/               # PIE families for reconstruction (ePIE / mPIE ...)
├── RPI_sims/               # RZP simulation setup and generation (vis / X-ray)
├── wave_propagation_cupy/  # wave propagation, Fourier padding (CuPy / NumPy)
├── fourier_shell_corr/     # Fourier Shell Correlation
└── misc/                   # FFT wrappers, image processing, plotting
```
