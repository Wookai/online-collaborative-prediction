function [mae, n_obs_list] = national_mae(model, Y_test, t_test, population, X_vote_test, X_muni, n_runs, varargin)
    n_obs_list = get_obs();
    
    n_cases = length(n_obs_list);
    [D, N_test] = size(Y_test);
    
    mae = zeros(2, n_cases, N_test, n_runs);

    % always use same sequence
    setSeed(123);
    for r = 1:n_runs
        % define reveal order
        order = randperm(D);

        for n = 1:N_test
            y = Y_test(:, n);

            for o = 1:n_cases
                n_obs = n_obs_list(o);
                obs_idx = false(D, 1);
                test_idx = true(D, 1);

                obs_idx(order(1:n_obs)) = true;
                test_idx(order(1:n_obs)) = false;
                
                obs_mean = mean(y(obs_idx));
                y_hat = zeros(size(y));
                
                y_hat(obs_idx) = y(obs_idx);
                y_hat(test_idx) = model.predict(y - obs_mean, obs_idx, test_idx, X_vote_test(n, :), X_muni, varargin{:}) + obs_mean;

                national_pred = y_hat' * population / sum(population);
                
                mae(1, o, n, r) = abs(t_test(n) - national_pred);
                mae(2, o, n, r) = (t_test(n) > .5) == (national_pred > .5);
            end
        end
    end
end

