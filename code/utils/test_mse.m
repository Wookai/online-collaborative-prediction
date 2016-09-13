function [mse, n_obs_list] = test_mse(model, Y_test, X_vote_test, X_muni, n_runs, varargin)
    n_test_munis = 236; % 10% of regions
    n_obs_list = get_obs();
    
    n_cases = length(n_obs_list);
    [D, N_test] = size(Y_test);
    
    mse = zeros(n_cases, N_test, n_runs);

    % always use same sequence
    set_seed(123);
    for r = 1:n_runs
        % define reveal order
        order = randperm(D);

        for n = 1:N_test
            y = Y_test(:, n);

            for o = 1:n_cases
                n_obs = n_obs_list(o);
                obs_idx = false(D, 1);
                test_idx = false(D, 1);

                obs_idx(order(1:n_obs)) = true;
                test_idx(order((end - (n_test_munis - 1)):end)) = true;
                
                obs_mean = mean(y(obs_idx));
                y_hat = model.predict(y - obs_mean, obs_idx, test_idx, X_vote_test(n, :), X_muni, varargin{:}) + obs_mean;

                mse(o, n, r) = mean((y(test_idx) - y_hat).^2);
            end
        end
    end
end

