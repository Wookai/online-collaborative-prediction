classdef model_lin_v < model
    % LIN(v) model
    % z_dn = gamma_d^T x_n
    
    properties
       lambda;
       Beta;
    end
    
    methods
        function m = model_lin_v(hyp)
            % default hyperparameter options
            [m.lambda] = m.get_hyperparameters(hyp);
        end
        
        function [lambda] = get_hyperparameters(~, hyp)
            [lambda] = myProcessOptions(hyp, 'lambda', 1);
        end
        
        function name = get_name(m)
            name = 'LIN(v)';
        end
        
        function summary = get_params_summary(m)
            summary = sprintf('lambda=%.15g', m.lambda);
        end
        
        function suffix = get_filename_suffix(m)
            suffix = sprintf('lin_v_lambda=%.15g', m.lambda);
        end
        
        function [train_rmse, valid_rmse] = fit(m, Y, train_idx, valid_idx, opts, varargin)
            n_var_args = length(varargin);
            
            assert(numel(varargin) >= 1, sprintf('%s needs vote features for training', m.get_name()));
            X_v = varargin{1};
            
            assert(isequal(size(Y), size(train_idx)), ...
                'Train indices should be the same for all votes');
            assert(isequal(size(Y), size(valid_idx)), ...
                'Valid indices should be the same for all votes');
            
            verbose = myProcessOptions(opts, 'verbose', 0);
            
            % initialize
            [D, N] = size(Y);
            M = size(X_v, 2);
            tX = [ones(N,1) X_v];
            Lambda = blkdiag(0, m.lambda*eye(M));
            
            m.Beta = zeros(M + 1, D);
            
            valid_rmse = 0;
            train_rmse = 0;

            for d = 1:D
                % get trainig and validation data
                Y_tr = Y(d, train_idx(d, :));
                Y_va = Y(d, valid_idx(d, :));
                tX_tr = tX(train_idx(d, :), :);
                tX_va = tX(valid_idx(d, :), :);

                % ridge reg
                K = (tX_tr'*tX_tr + Lambda);
                m.Beta(:, d) = K \ (tX_tr'*Y_tr');
                
                % error
                train_rmse = train_rmse + sum((Y_tr' - tX_tr * m.Beta(:, d)).^2);
                valid_rmse = valid_rmse + sum((Y_va' - tX_va * m.Beta(:, d)).^2);
            end

            % training rmse
            N_train = sum(train_idx(:));
            train_rmse = sqrt(train_rmse/N_train);
            
            % validation rmse
            N_valid = sum(valid_idx(:));
            valid_rmse = sqrt(valid_rmse/N_valid);
            
            if verbose
                fprintf('%f\t%f\n', train_rmse, valid_rmse);
            end
        end     
    
        function y_hat = predict(m, y, obs_idx, test_idx, varargin)
            assert(isequal(size(y), size(obs_idx)), ...
                'Mask and data should be the same size');
            assert(isequal(size(y), size(test_idx)), ...
                'Mask and data should be the same size');
            
            assert(numel(varargin) >= 1, sprintf('%s needs vote features for prediction', m.get_name()));
            X_v = varargin{1};
            
            N = size(X_v, 1);
            tX = [ones(N,1) X_v];
            
            y_hat = m.Beta(:, test_idx)' * tX';
        end
    end
end


