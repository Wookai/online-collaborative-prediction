classdef model_bias < model
    % BIAS model
    % z_dn = \mu_n

    methods
        function [train_log_lik, valid_rmse] = fit(~, Y, train_idx, valid_idx, opts, varargin)
            [verbose] = myProcessOptions(opts, 'verbose', 1);
            
            % compute the validation RMSE
            [~, N] = size(Y);
            
            train_log_lik = 0; valid_rmse = 0;
            for n = 1:N
              % training
              train_mean = mean(Y(train_idx(:, n)));
              train_log_lik = train_log_lik + sum((Y(train_idx(:, n)) - train_mean).^2);
              
              % validation
              valid_rmse = valid_rmse + sum((Y(valid_idx(:, n)) - train_mean).^2);
            end
            
            % training rmse
            N_train = sum(train_idx(:));
            train_log_lik = sqrt(train_log_lik/N_train);
            
            % validation rmse
            N_valid = sum(valid_idx(:));
            valid_rmse = sqrt(valid_rmse/N_valid);

            if verbose
                fprintf('%.4f %.4f\n', train_log_lik, valid_rmse);
            end
        end
        
        function y_hat = predict(~, y, obs_idx, test_idx, varargin)
            % compute the mean of observed elements
            vote_mean = mean(y(obs_idx));

            % predict this mean for all test elements
            y_hat = ones(size(y)) * vote_mean;
            y_hat = y_hat(test_idx);
        end
        
        function name = get_name(m)
            name = 'BIAS';
        end
        
        function suffix = get_filename_suffix(m)
            suffix = 'bias';
        end
    end
    
end

