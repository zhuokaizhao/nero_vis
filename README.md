# NERO: Non-Equivariance Revealed on Orbits

Code Repository of our NERO Plots paper

## Getting Started

A virtual environment is recommended for installing dependencies, e.g.,
using `conda`:

`conda create --name nero_env python=3.9`

and then

`conda activate nero_env`

Next we can install all the dependencies, which are summarized in `qt_app/setup_env.sh`.

`bash qt_app/setup_env.sh`

## NERO Interface

After installation, you should be able to run the NERO Interface by

`python qt_app/nero_app.py --mode digit_recognition --demo`

`nero_app.py` takes three arguments:

- `--mode`: Initialize the interface for different use cases. Currently it supports `digit_recognition`, `object_detection` and `piv`. But more to come. Please feel free to create a pull request for new interface.

- `--cache_path`: NERO Interface does computations in realtime and saves the results to `cache_path` that could be loaded next time during initialization. It saves time when users want to re-examine NERO plots that they created before. Can leave as open by default, but can also define a specific path that leads to a specific cache. One example use case could be that you have different versions of the same model that work all in one mode (`digit_recognition`, `object_detection` and `piv`).

- `--demo`: A binary flag that defines the behavior of NERO Interface. Without the flag, NERO Interface will be running in developing fashion that helps debugging. Users should always include this flag.

## Digit Recognition

As demoed in the Method section in our paper, digit recognition task can be visualized within NERO Interface by running

`python qt_app/nero_app.py --mode digit_recognition --demo`
