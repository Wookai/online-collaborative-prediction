import argparse

import plot_utils as pu

def main(args):
    n_runs = 10
    models = {
        'BIAS': [':', 'results_bias_nruns=%d.mat' % (n_runs,)],
        'LIN(r)': ['-.', 'results_lin_r_lambda=34_nruns=%d.mat' % (n_runs,)],
        'LIN(v)': ['--', 'results_lin_v_lambda=32_nruns=%d.mat' % (n_runs,)],
        'LIN(r) + LIN(v)': ['-', 'results_mf_lin_r_lin_v_L=0_lambdaU=1_lambdaV=1_biasU=1_biasV=1_featU=1_featV=1_lambdaBU=36_lambdaBV=80_nruns=%d.mat' % (n_runs,)],
    }
    order = ['BIAS', 'LIN(r)', 'LIN(v)', 'LIN(r) + LIN(v)']

    pu.plot_models_results(models, order, pu.get_rmse, args.save, 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    main(args)
