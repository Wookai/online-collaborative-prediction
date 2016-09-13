import argparse

import plot_utils as pu


def main(args):
    n_runs = 10
    models = {
        'LIN(v)': ['--', 'results_lin_v_lambda=32_nruns=%d.mat' % (n_runs,)],
        'MF + GP(r)': ['-.', 'results_mf_gp_r_seard_L=25_nruns=%d.mat' % (n_runs,)],
        'MF + GP(r) + LIN(v)': ['-', 'results_mf_gp_r_lin_v_seard_L=25_lambda=200_nruns=%d.mat' % (n_runs,)],
    }
    order = ['LIN(v)', 'MF + GP(r)', 'MF + GP(r) + LIN(v)']

    pu.plot_models_results(models, order, pu.get_rmse, args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    main(args)
