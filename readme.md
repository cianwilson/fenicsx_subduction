# Overview

Authors: Cian Wilson, Cameron Seebeck, Kidus Teshome, Nathan Sime, Peter van Keken

Welcome to the [_FEniCS Subduction Zone_ Jupyter Book](https://cianwilson.github.io/fenicsx_subduction), an online resource for modeling subduction zones!

This repository was developed by undergraduate interns Kidus Teshome and Cameron Seebeck at the [Carnegie Science Earth & Planets Laboratory](https://epl.carnegiescience.edu).  It is based on [Wilson & van Keken, PEPS, 2023 (II)](http://dx.doi.org/10.1186/s40645-023-00588-6), which is part II of a three part introductory review of the thermal structure of subduction zones by Peter van Keken and Cian Wilson.

Our goal is both to demonstrate how to build kinematic-slab thermal models of subduction zones using the finite element library [FEniCSx](https://fenicsproject.org) and to provide an easily accessible and modifiable source of the global suite of subduction zone models as described in [Wilson & van Keken, PEPS, 2023 (II)](http://dx.doi.org/10.1186/s40645-023-00588-6) and [van Keken & Wilson, PEPS, 2023 (III)](https://doi.org/10.1186/s40645-023-00589-5).  For comparison, the original models used in these papers are also available as open-source repositories on [github](https://github.com/cianwilson/vankeken_wilson_peps_2023) and [zenodo](https://doi.org/10.5281/zenodo.7843967).

# Usage

The notebooks in this repository can all be run interactively in a web browser, either online or locally.  In addition we provide versions of most notebooks as python scripts that can be run locally from a terminal.

## Online

The [website](https://cianwilson.github.io/fenicsx_subduction) is published as a [Jupyter Book](https://jupyterbook.org/). Each page
is a Jupyter notebook that can be run interactively in the browser.  To start such an interactive session using
[binder](https://mybinder.org/) click the ![Binder symbol](notebooks/images/binder.png)-symbol in the top right corner of the relevant page.  Note that [binder](https://mybinder.org/) may take some time to start.

```{admonition} Interactive Changes
Binder allows users to run notebooks interactively, making changes to the published Jupyter Book.  Note that these changes will be lost once the binder session ends unless the changed files are manually downloaded by selecting them in the Jupyter lab file browser and downloading them to the local machine.
```

```{admonition} Computational Costs
Binder limits the amount of computational resources available.  Extremely high resolution simulations may therefore not be feasible online.
```

## Local

To run the notebooks locally, outside of [binder](https://mybinder.org/), an installation of the FEniCSx is required. We strongly recommend new users do this using [Docker](https://www.docker.com/).

### Docker

Docker is software that uses _images_ and _containers_ to supply virtualized installations of software across different kinds of operating systems (Linux, Mac, Windows).  The first step is to install docker, following the instructions at their [webpage](https://docs.docker.com/get-started/).

Once docker is installed we provide compatible docker images using [github packages](https://github.com/users/cianwilson/packages/container/package/fenicsx_subduction).

```{admonition} Computational Resources
On non-linux operating systems docker limits the computational resources available to the virtualized docker container, which may limit the size of simulations it is possible to run locally.  Modify the docker settings to change these settings and make more resources available.
```

To use these images with this book on a local machine, first (using a terminal) clone the repository and change into that directory:
```bash
  git clone -b release https://github.com/cianwilson/fenicsx_subduction.git
  cd fenicsx_subduction
```

#### Browser

If running the book in a browser then run the following docker command:

```bash
  docker run --init --rm -p 8888:8888 --workdir /root/shared -v "$(pwd)":/root/shared ghcr.io/cianwilson/fenicsx_subduction:release
```
The first time this is run it will automatically download the docker image and start Jupyter lab in the docker container on the local machine.  To view the notebooks and modify them locally, copy and paste the URL printed in the terminal into a web-browser.

```{admonition} Updates
`docker run` will only download the docker image the first time it is called.  To get updates to the images run:

   docker pull ghcr.io/cianwilson/fenicsx_subduction:release

before calling `docker run`.
```

#### Terminal

Alternatively, the image can be used through an interactive terminal by running:

```bash
  docker run -it --rm -p 8888:8888 --workdir /root/shared -v "$(pwd)":/root/shared  --entrypoint="/bin/bash" ghcr.io/cianwilson/fenicsx_subduction:release
```

This allows the python scripts based on the notebooks to be run, e.g.:
```bash
  cd notebooks
  python3 <script name>.py
```
where `<script name>.py` should be substituted for the desired python script.

These can also be run in parallel using, e.g.:
```bash
  cd notebooks
  mpirun -np <p> python3 <script name>.py
```
where `<p>` should be substituted for the number of processes (see note above about computational resources in docker on non-linux operating systems).

Jupyter lab can also be started from within the docker container:
```bash
  jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```
again copying and pasting the resulting URL into a web browser to access the notebooks.

### Install

If not using docker a local installation of FEniCSx is necessary, including all of its components:
 * [UFL](https://github.com/FEniCS/ufl)
 * [Basix](https://github.com/FEniCS/basix)
 * [FFCx](https://github.com/FEniCS/ffcx)
 * [DOLFINx](https://github.com/FEniCS/dolfinx)
along with other dependencies, which can be seen in the files `docker/pyproject.toml` and `docker/Dockerfile`.

Installation instructions for FEniCSx are available on the [FEniCS project homepage](https://fenicsproject.org/download/).

## Acknowledgments

This Jupyter Book is based on the [FEniCSx Tutorial](https://jsdokken.com/dolfinx-tutorial/) by [Jørgen S. Dokken](https://jsdokken.com/).

