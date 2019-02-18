import os
import numpy as np
import pandas as pd
from pathlib import Path
import GPyOpt
from datetime import datetime
from tgp.summed_kernel import SummedKernel
from tgp.rbf_kernel import RBFKernel
from tgp.gp_predictor import GPPredictor
from tgp.multiplied_kernel import MultipliedKernel
from tgp.matern_kernels import MaternKernel32, MaternKernel12
from collections import namedtuple

ModelInputs = namedtuple('ModelInputs',
                         'winners losers days_since_start covariates')
ExperimentVars = namedtuple('ExperimentVars',
                            'kernel_fun bounds uses_surface')


def get_dataset():

    exec_dir = Path(os.path.abspath(__file__)).parents[1]
    csv_path = os.path.join(str(exec_dir), 'data', 'tennis_data.csv')
    return pd.read_csv(csv_path)


def get_experiment_data(start_date, end_date, add_surface=False):

    df = get_dataset()

    df = df[df['start_date'] >= start_date]
    df = df[df['start_date'] < end_date]

    winners = df['winner'].values
    losers = df['loser'].values
    days_since_start = (df['start_date'] -
                        df['start_date'].min()).dt.days.values

    if add_surface:
        surface = get_surface_covariates(df)
    else:
        surface = None

    return ModelInputs(winners=winners, losers=losers,
                       days_since_start=days_since_start,
                       covariates=surface)


def get_surface_covariates(df):

    # Make the covariates (surface)
    covariates = df['surface']

    # One-hot-encode
    mapping = {'clay': 0, 'grass': 1, 'hard': 2, 'indoor_hard': 3}
    cov_nums = [mapping[x] for x in covariates.values]
    covariate_array = np.zeros((covariates.shape[0], len(mapping)))
    covariate_array[np.arange(covariate_array.shape[0]), cov_nums] = 1

    return covariate_array


def to_minimise(theta, kernel_fun, model_inputs):

    # Deal with GPy weirdness of always having one extra dim
    theta = theta[0]

    kernel = kernel_fun(theta)
    predictor = GPPredictor(kernel)
    predictor.fit(model_inputs.winners, model_inputs.losers,
                  model_inputs.days_since_start,
                  covariates=model_inputs.covariates)

    neg_marg_lik = -predictor.calculate_log_marg_lik()

    print(neg_marg_lik)

    return neg_marg_lik


def get_single_matern():

    bounds = [{'name': 'l', 'type': 'continuous', 'domain': (0.1, 10.)},
              {'name': 'sd', 'type': 'continuous', 'domain': (0.01, 2.)}]

    def make_kernel(theta):

        return MaternKernel32(np.array([theta[0]]), theta[1])

    return ExperimentVars(kernel_fun=make_kernel, bounds=bounds,
                          uses_surface=False)


def get_two_matern():

    bounds = [{'name': 'l1', 'type': 'continuous', 'domain': (0.1, 10.)},
              {'name': 'l2', 'type': 'continuous', 'domain': (0.1, 10.)},
              {'name': 'sd1', 'type': 'continuous', 'domain': (0.01, 2.)},
              {'name': 'sd2', 'type': 'continuous', 'domain': (0.01, 2.)}]

    def make_kernel(theta):

        assert(len(theta) == 4)

        lscales = theta[:2]
        sds = theta[2:]

        kernels = list()
        kernels.append(MaternKernel12(np.array([lscales[0]]), sds[0]))
        kernels.append(MaternKernel32(np.array([lscales[1]]), sds[1]))
        kernel = SummedKernel(kernels)

        return kernel

    return ExperimentVars(kernel_fun=make_kernel, bounds=bounds,
                          uses_surface=False)


def get_matern_plus_surface():

    n_surf = 4

    bounds = [{'name': 'l_t', 'type': 'continuous', 'domain': (0.1, 20.)},
              {'name': 'sd_t', 'type': 'continuous', 'domain': (0.01, 2.)},
              {'name': 'ls_clay', 'type': 'continuous', 'domain': (0.1, 5.)},
              {'name': 'ls_grass', 'type': 'continuous', 'domain': (0.1, 5.)},
              {'name': 'ls_hard', 'type': 'continuous', 'domain': (0.1, 10.)},
              {'name': 'ls_indoor_hard', 'type': 'continuous',
              'domain': (0.1, 10.)}]

    def make_kernel(theta):

        assert(len(theta) == 2 + n_surf)

        kernels = list()

        # Time kernel
        kernels.append(MaternKernel32(np.array([theta[0]]), theta[1],
                                      active_dims=[0]))

        # Surface kernel
        kernels.append(RBFKernel(np.array(theta[2:n_surf+2]), 1.,
                                 active_dims=[1, 2, 3, 4]))

        kernel = MultipliedKernel(kernels)

        return kernel

    return ExperimentVars(kernel_fun=make_kernel, bounds=bounds,
                          uses_surface=True)


def get_optimiser(experiment_fun, start_date, end_date=datetime(2017, 12, 31),
                  method='gpy'):

    assert method in ['gpy', 'random']

    experiment_vars = experiment_fun()

    uses_surface = experiment_vars.uses_surface
    kernel_fun = experiment_vars.kernel_fun
    bounds = experiment_vars.bounds

    data = get_experiment_data(start_date=start_date, end_date=end_date,
                               add_surface=uses_surface)

    def curried_to_minimise(theta):

        return to_minimise(theta, kernel_fun, data)

    if method == 'gpy':
        return get_gpy_function(curried_to_minimise, bounds)
    else:
        return get_random_search_function(curried_to_minimise, bounds)


def get_gpy_function(curried_to_minimise, bounds):

    problem = GPyOpt.methods.BayesianOptimization(curried_to_minimise, bounds)

    def perform_gpy_opt(n_iter):

        problem.run_optimization(n_iter)

        return problem.x_opt, problem.fx_opt

    return perform_gpy_opt


def get_random_search_function(curried_to_minimise, bounds):

    def perform_random_search(n_iter):

        best_y = None
        best_x = None

        # Add the initial search too to make it fair
        for cur_iter in range(n_iter + 5):

            # Generate an x using the bounds
            cur_x = list()

            for cur_var in bounds:
                cur_domain = cur_var['domain']
                # Generate a random x
                random_var = np.random.uniform(cur_domain[0], cur_domain[1])
                cur_x.append(random_var)

            cur_x = np.array([cur_x])

            # Get the result
            cur_y = curried_to_minimise(cur_x)

            if best_y is None or cur_y < best_y:
                best_y = cur_y
                best_x = cur_x

        return best_x, best_y

    return perform_random_search
