import numpy as np
import GPyOpt
import pickle as pkl
from tgp.gp_predictor import GPPredictor
from tgp.summed_kernel import SummedKernel
# from tgp.rbf_kernel import RBFKernel
from tgp.matern_kernels import MaternKernel32, MaternKernel12
# from tgp.brownian_kernel import BrownianKernel
from tdata.datasets.oncourt_dataset import OnCourtDataset


# Let's try 3 RBF kernel components.
dataset = OnCourtDataset()
df = dataset.get_stats_df()

one_year = df[df['year'] >= 2016]
one_year = one_year[one_year['year'] < 2018]

# Prepare the inputs
winners = one_year['winner'].values
losers = one_year['loser'].values
days_since_start = (one_year['start_date'] -
                    one_year['start_date'].min()).dt.days.values

n_matches = winners.shape[0]

n_kerns = 2


def to_minimise(theta):

    theta = theta[0]

    lscales = theta[:n_kerns]
    sds = theta[n_kerns:]

    print(lscales)
    print(sds)

    kernels = list()

    # kernels.append(BrownianKernel(sds[0]))

    kernels.append(MaternKernel12(np.array([lscales[0]]), sds[0]))
    kernels.append(MaternKernel32(np.array([lscales[1]]), sds[1]))

    kernel = SummedKernel(kernels)
    predictor = GPPredictor(kernel)
    predictor.fit(winners, losers, days_since_start)

    neg_marg_lik = -predictor.calculate_log_marg_lik()

    print(neg_marg_lik)

    return neg_marg_lik


bounds = [{'name': 'l1', 'type': 'continuous', 'domain': (0.1, 10.)},
          {'name': 'l2', 'type': 'continuous', 'domain': (0.1, 10.)},
          {'name': 'sd1', 'type': 'continuous', 'domain': (0.01, 2.)},
          {'name': 'sd2', 'type': 'continuous', 'domain': (0.01, 2.)}]

max_iter = 100

# result = minimize(to_minimise, theta_init, tol=1)
problem = GPyOpt.methods.BayesianOptimization(to_minimise, bounds)
result = problem.run_optimization(max_iter)

pkl.dump({'best_x': problem.x_opt, 'best_y': problem.fx_opt, 'data': one_year},
         open('gpyopt_results_logit.pkl', 'wb'))
