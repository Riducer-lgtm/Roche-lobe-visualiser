
# Roche Lobe & Binary Mass Transfer Simulation

A Python (NumPy + Matplotlib) simulation of mass transfer in a close binary star system. It works in the co-rotating frame and visualises the Roche potential, the Lagrange points, and a gas stream flowing from the donor star through L1 onto the accretor.

## Features

- Effective (Roche) potential contours with the Roche lobe boundary
- Lagrange points L1–L5 (L1–L3 by bisection, L4/L5 analytically)
- Co-rotating frame dynamics: gravity, Coriolis and centrifugal forces
- Particle-based gas stream with RK4 integration
- Mass transfer between donor and accretor, with Roche lobe detachment
- Jacobi constant drift as a numerical accuracy diagnostic

## Files
| `Roche_Tolerance.py` | Single test particle in an equal-mass binary; Roche lobe found from the potential gradient |
| `Roche_Lobe_Optimized.py` | Roche potential, L1–L5 via bisection, leapfrog test particle with trail |
| `mass_transfer_prototype.py` | Multi-particle leapfrog prototype of the gas stream |
| `Mass_transfer.py` | Intermediate development version |
| `Mass_Transfer_Final.py` | Full simulation: RK4, mass transfer, donor detachment, live potential updates |

## Usage
'''
pip install numpy matplotlib
python Mass_Transfer_Final.py

'''

Parameters (masses, particle count, timestep, capture radius) are set at the top of each script.

## Roadmap

- GPU-accelerated version
- General-relativistic effects

## Author

Har Ridam Singh, BS-MS, IISER Pune

MAIL-'har.ridamsingh@students.iiserpune.ac.in'
