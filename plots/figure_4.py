import argparse

import plot_utils as pu


def main(args):
    n_runs = 10
    models = {
        'BIAS': [':', 'national_bias_nruns=%d.mat' % (n_runs,)],
        'LIN(v)': ['-.', 'national_lin_v_lambda=32_nruns=%d.mat' % (n_runs,)],
        'MF + GP(r)': ['--', 'national_mf_gp_r_seard_L=25_nruns=%d.mat' % (n_runs,)],
        'MF + GP(r) + LIN(v)': ['-', 'national_mf_gp_r_lin_v_seard_L=25_lambda=200_nruns=%d.mat' % (n_runs,)],
    }
    order = ['BIAS', 'LIN(v)', 'MF + GP(r)', 'MF + GP(r) + LIN(v)']

    pu.plot_models_results(models, order, pu.get_mae, args.save, national=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save')

    args = parser.parse_args()
    main(args)
