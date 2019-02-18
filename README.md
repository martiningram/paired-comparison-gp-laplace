# Dynamic paired comparison modelling with Gaussian Processes

This repository contains code to fit dynamic paired comparison models using
Gaussian Process priors.

## Requirements

The minimum requirements to use the library are listed in `requirements.txt`.
You can install them by running:

```bash
pip install -r requirements.txt
```

Note that `scikit-sparse` requires that the `SuiteSparse` library is installed.
This dependency is most easily handled with anaconda, where you can do:

```bash
conda install suitesparse
```

If that doesn't work for you, there are other instructions here: [scikit-sparse
documentation](https://scikit-sparse.readthedocs.io/en/latest/overview.html#requirements)

If you want to run the demo notebooks, you will also have to install the
requirements in `demo_requirements.txt`:

```bash
pip install -r demo_requirements.txt
```

Once the requirements are installed, you can run:

```bash
python setup.py install
```

To install the library for you.

## How to use

The easiest way to get started is to view the demos in the `jupyter` folder.

* `Time only demo.ipynb` fits a Matern 3/2 kernel to tennis data, shows the
  inferred latent functions, and has a prediction example.
* `Surface demo.ipynb` fits a Matern 3/2 kernel on time multiplied with an ARD
  RBF kernel on surface, shows the latent functions, and has a prediction
  example, too.
