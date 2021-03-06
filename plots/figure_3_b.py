import argparse

import plot_utils as pu


def main(args):
    n_runs = 10
    models = {
        'LIN(r)': ['-.', 'results_lin_r_lambda=34_nruns=%d.mat' % (n_runs,)],
        'MF': [':', 'results_mf_lin_r_lin_v_L=25_lambdaU=0.0316_lambdaV=31.6_biasU=1_biasV=1_featU=0_featV=0_lambdaBU=0.01_lambdaBV=10_nruns=%d.mat' % (n_runs,)],
        'MF + LIN(r) (CV)': ['--', 'results_mf_lin_r_lin_v_L=25_lambdaU=0.08_lambdaV=28_biasU=1_biasV=1_featU=0_featV=1_lambdaBU=100_lambdaBV=10_nruns=%d.mat' % (n_runs,)],
        'MF + LIN(r) (hand)': ['-', 'results_mf_lin_r_lin_v_L=25_lambdaU=0.0316_lambdaV=31.6_biasU=1_biasV=1_featU=0_featV=1_lambdaBU=34_lambdaBV=10_nruns=%d.mat' % (n_runs,)],
    }
    order = ['LIN(r)', 'MF', 'MF + LIN(r) (CV)', 'MF + LIN(r) (hand)']

    pu.plot_models_results(models, order, pu.get_rmse, args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    main(args)
