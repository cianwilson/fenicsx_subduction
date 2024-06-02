# Overview

The thermal structure of subduction zones is fundamental to our understanding of the physical and chemical processes that occur at active convergent plate margins.  These include magma generation and related arc volcanism, shallow and deep seismicity and metamorphic reactions that can release fluids.  Computational models can predict the thermal structure to great numerical precision when models are fully described but this does not guarantee accuracy or applicability.  In a trio of companion papers ([I](http://dx.doi.org/10.1186/s40645-023-00573-z), [II](http://dx.doi.org/10.1186/s40645-023-00588-6), and [III](http://dx.doi.org/10.1186/s40645-023-00589-5)), the construction of thermal subduction zone models, their use in subduction zone studies, and their link to geophysical and geochemical observations were explored.  This [Jupyter Book](https://jupyterbook.org/) reproduces the examples presented in part [II](http://dx.doi.org/10.1186/s40645-023-00588-6) where the finite element techniques that can be used to predict the thermal structure were discussed in an introductory fashion along with their verification and validation.  Unlike the examples presented in part [II](http://dx.doi.org/10.1186/s40645-023-00588-6) ([github](https://github.com/cianwilson/vankeken_wilson_peps_2023), [zenodo](https://doi.org/10.5281/zenodo.7843967)), which mostly used the finite element model builder [TerraFERMA](https://terraferma.github.io) and a legacy version of the finite element library [FEniCS](https://fenicsproject.org/download/archive/), here we present these models using the latest version of FEniCS, [FEniCSx](https://fenicsproject.org/).

## Interactive tutorials

As this book has been published as a [Jupyter Book](https://jupyterbook.org/), we provide interactive notebooks that can be run in the browser. To start such a notebook click the ![Binder symbol](notebooks/images/binder.png)-symbol in the top right corner of the relevant tutorial.

## Obtaining the software

If you would like to work with FEniCSx outside of the binder-notebooks, you need to install the FEniCS software.  The recommended way of installing DOLFINx for new users is by using Docker.
Docker is a software that uses _containers_ to supply software across different kinds of operating systems (Linux, Mac, Windows). The first step is to install docker, following the instructions at their [webpage](https://docs.docker.com/get-started/).

### Docker images

Compatible images are available in the [Github Packages](https://github.com/users/cianwilson/packages/container/package/fenicsx_subduction).

To use the notebooks in this tutorial with DOLFINx on your own computer, you should (using a terminal) clone the repository and change into that directory:
```bash
git clone https://github.com/cianwilson/fenicsx_subduction.git
cd fenicsx_subduction
```
Then run the following docker command:

```bash
  docker run --init --rm -p 8888:8888 --workdir /root/shared -v "$(pwd)":/root/shared ghcr.io/cianwilson/fenicsx_subduction:v0.8.0
```
Finally, copy and paste the URL printed in the terminal into a web-browser.

Alternatively, this image can used as an interactive terminal by running:

```bash
  docker run -it --rm -p 8888:8888 --workdir /root/shared -v "$(pwd)":/root/shared  --entrypoint="/bin/bash" ghcr.io/cianwilson/fenicsx_subduction:v0.8.0
```
Jupyter lab can then be started from within the docker container:
```bash
  jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```
again copying and pasting the resulting URL into a web browser.

## Acknowledgments

This Jupyter Book is based on the [FEniCSx Tutorial](https://jsdokken.com/dolfinx-tutorial/) by [Jørgen S. Dokken](https://jsdokken.com/). 
