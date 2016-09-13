classdef model < handle
    %MODEL Abstract class defining the common elements of models.
    
    methods(Abstract)
        % Fit the model m to the given training data Y, indicated by the indicator matrix train_idx.
        % The training error (RMSE or negative loglikelihood) is returned, along with the validation
        % RMSE computed on the entries indicated by valid_idx.
        [train_error, valid_rmse] = fit(m, Y, train_idx, valid_idx, options, varargin);
        
        % Predict the elements of y identified by test_idx, when observing
        % the elements identified by obs_idx.
        y_hat = predict(m, y, obs_idx, test_idx, varargin);
        
        % Get the name of this model
        name = get_name(m);
    end
    
    methods
        
        function summary = get_params_summary(m)
            % Get a string resuming all the important param values
            summary = 'N/A';
        end
        
        function suffix = get_filename_suffix(m)
            % Get a string suffix that summarises the params for this
            % model, for appending to a filename when saving.
            
            suffix = '';
        end
        
        function rmse = copmute_rmse(m, Y, obs_idx, test_idx, varargin)
                N = size(Y, 2);
                rmse = 0;
                n_samples = 0;
                
                if numel(varargin) > 0
                    X_v = varargin{1};
                else
                    X_v = zeros(N, 0);
                end
                
                for n = 1:N
                    y = Y(:, n);

                    y_hat = m.predict(y, obs_idx(:, n), test_idx(:, n), X_v(n, :), varargin{2:end});

                    err = (y(test_idx(:, n)) - y_hat).^2;
                    rmse = rmse + sum(err);
                    n_samples = n_samples + sum(test_idx(:, n));
                end

                rmse = sqrt(rmse / n_samples);
        end
    end
    
end

