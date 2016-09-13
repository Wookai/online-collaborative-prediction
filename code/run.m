% Fit all models and compute the results on test votes

clear all;
set_path;

%%%%%%%%%%%%
num_test_votes = 50; % number of votes kept for testing
n_test_runs = 10; % number of runs 
prop_valid = 0.1; % proportion of training data randomly selected as validation

opt.max_iters = 100;
opt.tol = 1e-5;
opt.verbose = 1;
%%%%%%%%%%%%

[Y, t, X_vote, X_muni, population] = load_data();

% create train-test using past votes as train and new as test
[D, N] = size(Y);

Y_train = Y(:, 1:(N - num_test_votes));
X_vote_train = X_vote(1:(N - num_test_votes), :);

Y_test = Y(:, (N - num_test_votes + 1):end);
X_vote_test = X_vote((N - num_test_votes + 1):end, :);
t_test = t((N - num_test_votes + 1):end); % national result

% center training data
Y_train = bsxfun(@minus, Y_train, mean(Y_train));

% create train-validation split by punching holes
N_train = size(Y_train, 2);
num_valid_entries = round(prop_valid*D);
train_idx = true(D, N_train);
valid_idx = false(D, N_train);

set_seed(123);
for n = 1:N_train
  idx = randperm(D);
  idx_valid = idx(1:num_valid_entries);
  train_idx(idx_valid, n) = false;
  valid_idx(idx_valid, n) = true;
end

models = {
    {'BIAS', model_bias()}
    {'LIN(r)', model_lin_r(struct('lambda', 34))}
    {'LIN(v)', model_lin_v(struct('lambda', 32))}
    {'LIN(r) + LIN(v)', model_mf_lin_r_lin_v(struct('L', 0, 'featU', 1, 'featV', 1, 'lambdaU', 1, 'lambdaV', 1, 'lambdaBU', 36, 'lambdaBV', 80, 'biasU', 1, 'biasV', 1))}
	{'MF', model_mf_lin_r_lin_v(struct('L', 25, 'featU', 0, 'featV', 0, 'lambdaU', 0.0316, 'lambdaV', 31.6, 'biasU', 1, 'biasV', 1))}
	{'MF + LIN(r) (CV)', model_mf_lin_r_lin_v(struct('L', 25, 'featU', 0, 'featV', 1, 'lambdaU', 0.08, 'lambdaV', 28, 'biasU', 1, 'biasV', 1, 'lambdaBU', 100))}
	{'MF + LIN(r) (hand)', model_mf_lin_r_lin_v(struct('L', 25, 'featU', 0, 'featV', 1, 'lambdaU', 0.0316, 'lambdaV', 31.6, 'biasU', 1, 'biasV', 1, 'lambdaBU', 34))}
    {'MF + GP(r) (linear)', model_mf_gp_r_liniso(struct('L', 25))}
    {'MF + GP(r)', model_mf_gp_r_seard(struct('L', 25))}
    {'MF + GP(r) + LIN(v)',  model_gp_vu_reg_seard(struct('L', 25, 'lambda', 200))}
};

n_models = size(models, 1);
for m_id = 1:n_models;
    m_name = models{m_id}{1};
    m = models{m_id}{2};
        
    model_filename = sprintf('../models/%s.mat', m.get_filename_suffix());
    results_filename = sprintf('../results/results_%s_nruns=%d.mat', m.get_filename_suffix(), n_test_runs);
    national_results_filename = sprintf('../results/national_%s_nruns=%d.mat', m.get_filename_suffix(), n_test_runs);
    
    % try to load the model from file
    if exist(model_filename, 'file') == 2
        data = load(model_filename);
        m = data.model;
    else
        % if it does not exist, train and save it
        fprintf('Fitting %s model...\n', m_name);        
        m.fit(Y_train, train_idx, valid_idx, opt, X_vote_train, X_muni);
        
        fprintf('Saving %s model...\n', m_name);
        model = m;
        save(model_filename, 'model');
    end
    
    % compute results if they do not already exist
    if exist(results_filename, 'file') ~= 2
        fprintf('Computing %s test results...\n', m_name);
        [model_test_mse, obs] = test_mse(m, Y_test, X_vote_test, X_muni, n_test_runs);
        fprintf('Saving %s results...\n', m_name);
        save(results_filename, 'model_test_mse', 'obs');
    end

    if exist(national_results_filename, 'file') ~= 2
        fprintf('Computing %s national results...\n', m_name);
        model_test_mae = national_mae(m, Y_test, t_test, population, X_vote_test, X_muni, n_test_runs);

        fprintf('Saving %s results...\n', m_name);
        save(results_file, 'model_test_mae', 'obs');
    end
end