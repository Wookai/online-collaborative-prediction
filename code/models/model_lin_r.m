classdef model_lin_r < model
    % LIN(r) model
    % z_dn = beta_n^T x_d

    properties
       lambda;
    end
    
    methods
        function m = model_lin_r(hyp)
            % default hyperparameter options
            [m.lambda] = m.get_hyperparameters(hyp);
        end
        
        function [lambda] = get_hyperparameters(~, hyp)
            [lambda] = myProcessOptions(hyp, 'lambda', 1);
        end
        
        function name = get_name(m)
            name = 'LIN(r)';
        end
        
        function summary = get_params_summary(m)
            summary = sprintf('lambda=%.15g', m.lambda);
        end
        
        function suffix = get_filename_suffix(m)
            suffix = sprintf('lin_r_lambda=%.15g', m.lambda);
        end
        
        function [train_rmse, valid_rmse, gradient] = fit(m, Y, train_idx, valid_idx, ~, varargin)
            n_var_args = length(varargin);
            
            assert(numel(varargin) >= 2, sprintf('%s needs region features for training', m.get_name()));
            X = varargin{2};
           
            lambda_ = m.lambda;
            if n_var_args == 3
                [lambda_] = m.get_hyperparameters(varargin{3});
            end
            
            assert(isequal(size(Y), size(train_idx)), ...
                'Train indices should be the same for all votes');
            assert(isequal(size(Y), size(valid_idx)), ...
                'Valid indices should be the same for all votes');
            
            % initialize
            [D, N] = size(Y);
            M = size(X,2);
            tX = [ones(D,1) X];
            Lambda = blkdiag(0, lambda_*eye(M));
            
            valid_rmse = 0;
            train_rmse = 0;
            gradient = 0;

            for n = 1:N
                % get trainig and validation data
                Y_tr = Y(train_idx(:, n), n);
                Y_va = Y(valid_idx(:, n), n);
                tX_tr = tX(train_idx(:, n), :);
                tX_va = tX(valid_idx(:, n), :);

                % ridge reg
                K = (tX_tr'*tX_tr + Lambda);
                beta = K \ (tX_tr'*Y_tr);

                % error
                train_rmse = train_rmse + sum((Y_tr - tX_tr * beta).^2);
                valid_rmse = valid_rmse + sum((Y_va - tX_va * beta).^2);
                
                if nargout == 3
                    gradient = gradient + norm(beta, 2);
                end
            end

            % training rmse
            N_train = sum(train_idx(:));
            train_rmse = sqrt(train_rmse/N_train);
            
            % validation rmse
            N_valid = sum(valid_idx(:));
            valid_rmse = sqrt(valid_rmse/N_valid);
            
            gradient = gradient / N;
        end     
        
        function [f, df] = objective_function(vals, m, Y, train_idx, valid_idx, options, X)
            hyp = struct('lambda', vals(1));
            [f, ~, df] = m.fit(Y, train_idx, valid_idx, options, [], X, hyp);
        end
        
        function hyp_opt = learn(m, Y, train_idx, valid_idx, fit_options, X, hyp0, minFunc_options)
            lambda_opt = minFunc(@objective_function, hyp0, minFunc_options, m, Y, train_idx, valid_idx, fit_options, X);
            hyp_opt.lambda = lambda_opt;
        end
    
        function y_hat = predict(m, y, obs_idx, test_idx, varargin)
            assert(isequal(size(y), size(obs_idx)), ...
                'Mask and data should be the same size');
            assert(isequal(size(y), size(test_idx)), ...
                'Mask and data should be the same size');
            
            assert(numel(varargin) >= 2, sprintf('%s needs region features for prediction', m.get_name()));
            X = varargin{2};
            
            scaling = 1;
            if numel(varargin) >= 3
                scaling = varargin{3};
            end
            
            [D, M] = size(X);
            
            Lambda = blkdiag(0, scaling*m.lambda*eye(M));
            
            tX = [ones(D,1) X];
            X_obs = tX(obs_idx, :);
            X_test = tX(test_idx, :);
            
            beta = (X_obs'*X_obs + Lambda) \ (X_obs'*y(obs_idx));
            
            y_hat = X_test * beta;
        end
    end
end


