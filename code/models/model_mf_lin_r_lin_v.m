classdef model_mf_lin_r_lin_v < model
    % MF + LIN(r) + LIN(v) model
    % z_dn = v_d^T u_n + \beta_n^T x_d + \gamma_d^T x_n
    % By setting L, featU, or featV to 0, we can disable any of the 3 components,
    % to create MF + LIN(r) or LIN(r) + LIN(v)
    
    properties
       lambdaU;
       lambdaV;
       lambdaBU;
       lambdaBV;
       L;
       U;
       V;
       biasU;
       biasV;
       B_U;
       B_V;
       featU;
       featV;
    end
    
    methods
        function m = model_mf_lin_r_lin_v(hyp)
            % default hyperparameter options
            [m.L, m.lambdaU, m.lambdaV, m.biasU, m.biasV, m.featU, m.featV, m.lambdaBU, m.lambdaBV] = myProcessOptions(hyp, ...
                'L', 15, ...
                'lambdaU', 0.01, ...
                'lambdaV', 0.01, ...
                'biasU', 0, ...
                'biasV', 0, ...
                'featU', 0, ...
                'featV', 0, ...
                'lambdaBU', 0.01, ...
                'lambdaBV', 10);
        end
        
        function name = get_name(m)
            if m.L == 0
                name = 'LIN(r) + LIN(v)'
            else
                if m.featU == 0 && m.featV == 0
                    name = 'MF'
                elseif m.featU == 0 && m.featV == 1
                    name = 'MF + LIN(r)'
                elseif m.featU == 1 && m.featV == 0
                    name = 'MF + LIN(v)'
                else m.featU == 0 && m.featV == 1
                    name = 'MF + LIN(r) + LIN(v)';
                end
            end
        end
        
        function summary = get_params_summary(m)
            summary = sprintf('L=%d, lU=%.15g, lV=%.15g, bU=%d, bV=%d, fU=%d, fV=%d, lbU=%.15g, lbV=%.15g', ...
                              m.L, m.lambdaU, m.lambdaV, m.biasU, m.biasV, m.featU, m.featV, m.lambdaBU, m.lambdaBV);
        end
        
        function suffix = get_filename_suffix(m)
            suffix = sprintf('mf_lin_r_lin_v_L=%d_lambdaU=%.15g_lambdaV=%.15g_biasU=%d_biasV=%d_featU=%d_featV=%d_lambdaBU=%.15g_lambdaBV=%.15g', ...
                             m.L, m.lambdaU, m.lambdaV, m.biasU, m.biasV, m.featU, m.featV, m.lambdaBU, m.lambdaBV);
        end
        
        function [X_U, X_V] = get_features(m, varargin)
            % features are of size [N, F_U] and [D, F_V]
            X_U = [];
            X_V = [];
            if numel(varargin) > 0 && m.featU
                X_U = varargin{1};
            end
            if numel(varargin) > 1 && m.featV
                X_V = varargin{2};
            end 
        end
        
        
        function [train_rmse, valid_rmse] = fit(m, Y, train_idx, valid_idx, options, varargin)
            assert(isequal(size(Y), size(train_idx)), ...
                'Mask and data should be the same size');
            assert(isequal(size(Y), size(valid_idx)), ...
                'Mask and data should be the same size');
                        
            % parse options
            [max_iters, tol, verbose] = myProcessOptions(options, 'max_iters', 30, 'tol', 1e-5, 'verbose', 1);
            
            [D, N] = size(Y);
            
            [X_U, X_V] = m.get_features(varargin{:});

            % get number of features for each domain
            F_U = size(X_U, 2);
            F_V = size(X_V, 2);
            
            % initialize V
            if m.biasV
                m.V = [rand(D, m.L) ones(D, 1)];
            else
                m.V = rand(D, m.L);
            end
            
            % regression parameters are of size [N, F_V] and [D, F_U]
            m.B_V = rand(D, F_U);
            
            % take care of empty features
            if isempty(X_U)
                X_U = zeros(N, 0);
            end
            if isempty(X_V)
                X_V = zeros(D, 0);
            end 
            
            % ALS
            for iter = 1:max_iters                
                Y_no_features_U = Y - m.B_V * X_U';
                X_V_V = [X_V m.V];
                
                % update U
                B_U_U = m.update_U(Y_no_features_U, X_V_V, m.lambdaU, train_idx, m.biasU, m.biasV, F_V, m.lambdaBU);
                
                % separate U and beta_V
                m.B_U = B_U_U(1:F_V, :)';
                m.U = B_U_U((1 + F_V):end, :);
                
                % remove contribution from features of V
                Y_no_features_V = Y - X_V * m.B_U';
                X_U_Ut = [X_U m.U'];

                % update V
                B_V_Vt = m.update_U(Y_no_features_V', X_U_Ut, m.lambdaV, train_idx', m.biasV, m.biasU, F_U, m.lambdaBV);
                
                % separate V and beta_U
                m.B_V = B_V_Vt(1:F_U, :)';
                m.V = B_V_Vt((1 + F_U):end, :)';
                
                % extract bias
                [V_nobias, c, ~] = m.extract_bias(m.V, m.biasU, m.biasV);
                
                % duplicate column bias to add to each column, if needed
                if c ~= 0
                    c = repmat(c, 1, N);
                end
                
                Y_hat = V_nobias * m.U + c + X_V * m.B_U' + m.B_V * X_U';
                
                % compute train and validation rmse
                err_train = Y(train_idx) - Y_hat(train_idx);
                train_rmse(iter) = sqrt(mean(err_train.^2));
                
                err_valid = Y(valid_idx) - Y_hat(valid_idx);
                valid_rmse(iter) = sqrt(mean(err_valid.^2));

                if verbose
                    fprintf('%d %.4f %.4f\n', iter, train_rmse(iter), valid_rmse(iter));
                end

                % convergence
                converged = isConverged(train_rmse, tol, 'objFun');
                if converged; break; end;
            end
        end
        
        function y_hat = predict(m, y, obs_idx, test_idx, varargin)
            assert(isequal(size(y), size(obs_idx)), ...
                'Mask and data should be the same size');
            assert(isequal(size(y), size(test_idx)), ...
                'Mask and data should be the same size');
            
            [X_u, X_V] = m.get_features(varargin{:});
            
            D = size(m.V, 1);
            
            scaling = 1;
            scalingB = 1;
            if numel(varargin) >= 3
                scaling = varargin{3};
                if numel(varargin) >= 4
                    scalingB = varargin{4};
                end
            end
            
            % take care of empty features
            if isempty(X_u)
                X_u = zeros(1, 0);
            end
            if isempty(X_V)
                X_V = zeros(D, 0);
            end 
            
            F_V = size(X_V, 2);

            y_no_features_U = y - m.B_V * X_u';
            X_V_V = [X_V m.V];

            % update U
            B_u_u = m.update_U(y_no_features_U, X_V_V, m.lambdaU * scaling, obs_idx, m.biasU, m.biasV, F_V, m.lambdaBU * scalingB);
                
            % separate U and beta_V
            B_u = B_u_u(1:F_V, :)';
            u = B_u_u((1 + F_V):end, :);
            
            Vt = m.V(test_idx, :);            
            [Vt, ct, ~] = m.extract_bias(Vt, m.biasU, m.biasV);
            
            % duplicate column bias to add to each column, if needed
            if ct ~= 0
                ct = repmat(ct, 1, size(u, 2));
            end
            
            y_hat = Vt * u + ct + X_V(test_idx, :) * B_u' + m.B_V(test_idx, :) * X_u';
        end
        
        function U = update_U(m, Y, V, lambda, train_idx, biasU, biasV, n_feat, lambdaB)
            % Update u_n given v_i or Alternating least-squares for minimizing
            % (Y_in - (c_i + d_n + v_i'*u_n))^2
            % 
            % Y is data size DxN
            % V is D x (L+1) with [v_i' c_i] as rows
            %   when no bias then V is D x L with v_i as rows
            % lambda is the regularization parameter
            % biasU is 1, then dn is non-zero
            % biasV is 1, then ci is non-zero
            
            assert(isequal(size(Y), size(train_idx)), ...
                'Mask and data should be the same size');

            [~, N] = size(Y);

            L_ = size(V, 2);
            if biasV
                L_ = L_ - 1;
            end
            
            % init U
            if biasU
                U = zeros(L_ + 1, N);
            else
                U = zeros(L_, N);
            end

            % for all n
            for n = 1:N
                % observed training entries
                On = train_idx(:, n);

                % if there is some training data
                if ~isempty(On)
                    yo = Y(On, n);
                    Vo = V(On, :);
                    
                    [Vo, co, I] = m.extract_bias(Vo, biasU, biasV);
                    
                    % combine lambdas if needed, first comes regression
                    % coeffs, than latent features
                    Lambda = blkdiag(lambdaB * eye(n_feat), lambda * I((n_feat + 1):end, (n_feat + 1):end));
                    
                    % update un = inv(Vo'*Vo + lambda*I)
                    Hn = Vo'*Vo + Lambda;
                    un = Hn \ (Vo'* (yo - co));
                    U(:,n) = un;
                end
            end
        end
        
        function [V_nobias, c, I] = extract_bias(~, V_full, biasU, biasV)
            % Extract the bias from V and adds 1 if required
            
            [n_rows, L_] = size(V_full);
            if biasV
                L_ = L_ - 1;
            end
            
            if biasU && biasV
                V_nobias = [V_full(:, 1:end-1) ones(n_rows, 1)]; % extract V without bias, add 1s for U's bias
                c = V_full(:, end); % extract bias
                I = blkdiag(eye(L_), 0); % don't regularize the bias
            elseif biasU && ~biasV
                V_nobias = [V_full ones(n_rows, 1)]; % add 1s for U's bias
                c = 0;
                I = blkdiag(eye(L_), 0); % don't regularize the bias
            elseif ~biasU && biasV
                V_nobias = V_full(:, 1:end-1); % extract V without bias
                c = V_full(:, end); % extract bias
                I = eye(L_);
            else
                V_nobias = V_full;
                c = 0;
                I = eye(L_);
            end
        end
    end
    
end

