import os
import pickle
from tqdm import tqdm
from datetime import datetime
import tgp.experiments as tgpe


start_date = datetime(2016, 1, 1)
end_date = datetime(2017, 12, 31)

num_reruns = 10
max_iters = [10, 50, 100, 200]
optimiser = 'random'

for optimiser in ['gpy', 'random']:

    for exp_name, experiment_fun in zip(
            ['single_matern', 'two_matern', 'matern_plus_surf'],
            [tgpe.get_single_matern, tgpe.get_two_matern,
                tgpe.get_matern_plus_surface]):

        target_dir = (f'experiment_runs_{optimiser}_{exp_name}'
                      f'_{start_date}_{end_date}')

        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        run_time = datetime.now()

        with open(f'{target_dir}/result_stream_{run_time}.txt', 'w') as output_file:

            all_results = list()

            # Maybe look into joblib to run this more efficiently
            for cur_max_iter in tqdm(max_iters):
                for cur_rerun in range(num_reruns):
                    optim_fun = tgpe.get_optimiser(experiment_fun, start_date,
                                                   end_date, method=optimiser)
                    # Run optimisation
                    best_x, best_y = optim_fun(cur_max_iter)
                    all_results.append({
                        'best_x': best_x,
                        'best_y': best_y,
                        'max_iter': cur_max_iter
                    })
                    # Also write these to a file in a streaming fashion
                    output_file.write(
                        f'{datetime.now()} ; {cur_max_iter} ; {best_x} ; {best_y} \n')
                    output_file.flush()

        pickle.dump(all_results, open(f'{target_dir}/all_results_{run_time}.pkl',
                                      'wb'))
