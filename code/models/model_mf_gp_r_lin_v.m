classdef model_mf_gp_r_lin_v < handle
    % Base class for the MF + GP(r) + LIN(v) model
    % z_dn = v_d^T u_n + GP(m(x_d), k(x_d1, x_d2)) + gamma_d^T x_n
    % To use different mean and covariance functions, simply extend this class
    % and return the desired function in get_mean_cov

    properties
        lik;
        mean;
        cov;
        hyp;
        V;
        sn2;
        Beta;
    end
    
    properties (Transient = true)
        mu;
        Sigma;
    end
    
    methods(Abstract)
        [mean, cov] = get_mean_cov(m);
        [lik_hyp, mean_hyp, cov_hyp, lambda] = get_hyperparameters(m, hyp);
        % methods for learning
        hyp = vals_to_hyp(m, vals);
        vals = hyp_to_vals(m, hyp);
        d_cov_size = covariance_gradient_size(m);
    end
    
    methods
        function m = model_mf_gp_r_lin_v(hyp, V, sn2)
            m.lik = {@likGauss};
            [m.mean, m.cov] = m.get_mean_cov();
            [m.hyp.lik, m.hyp.mean, m.hyp.cov, m.hyp.L, m.hyp.lambda] = m.get_hyperparameters(hyp);
            
            if nargin >= 2
                m.V = V;
                if nargin >= 3
                    m.sn2 = sn2;
                end
            end
        end
      
        function [train_nlZ, valid_rmse] = fit(m, Y, ~, ~, options, varargin)
            assert(numel(varargin) >= 2, sprintf('%s needs region and vote features for training', m.get_name()));
            X_v = varargin{1};
            X_m = varargin{2};

            [verbose, max_iters, tol, epsilon] = myProcessOptions(options, 'verbose', 0, 'max_iters', 100, 'tol', 1e-3, 'epsilon', 1e-5);
            
            [D, N] = size(Y);
            m.sn2 = 1;
            m.V = rand(D, m.hyp.L);
            valid_rmse = NaN;
            
            % remove vote regression component from the data first
            M = size(X_v, 2);
            tX = [ones(N,1) X_v];
            m.Beta = (tX'*tX + blkdiag(0, m.hyp.lambda*eye(M))) \ (tX'*Y');
            Y = Y - m.Beta' * tX';
            
            m.hyp

            for it = 1:max_iters
                m.mu = feval(m.mean{:}, m.hyp.mean, X_m);
                K = feval(m.cov{:}, m.hyp.cov, X_m) + epsilon*eye(D);
                m.Sigma = m.sn2*K + m.V*m.V';
                
                obs_sn2 = exp(2*m.hyp.lik);
                if obs_sn2 < 1e-6
                    L_Sigma = chol(m.Sigma + obs_sn2*eye(D));
                    sl = 1;
                else
                    L_Sigma = chol(m.Sigma/obs_sn2 + eye(D));
                    sl = obs_sn2;
                end

                C = zeros(D, D);
                for n = 1:N
                    y = Y(:, n);
                    
                    alpha = solve_chol(L_Sigma, y - m.mu)/sl;
                    E_tn = m.mu + m.Sigma*alpha;
                    
                    C = C + E_tn*E_tn';
                end
                C = C / N;
                
                v = solve_chol(L_Sigma, m.Sigma)/sl;
                cov_tn = m.Sigma - m.Sigma*v;
                C = C + cov_tn;

                L_K = chol(K, 'lower');
                L_K_inv = inv(L_K')';

                C_tilde = L_K_inv * C * L_K_inv';
                [R, Lambda] = eigs(C_tilde, m.hyp.L);

                sn2_hat = (trace(C_tilde) - trace(Lambda))/(D - m.hyp.L);
                W_hat = R*(Lambda - sn2_hat*eye(m.hyp.L))^(1/2);
                
                m.V = L_K*W_hat;
                m.sn2 = sn2_hat;
                
                [m.hyp, train_nlZ(it)] = m.learn_gp(Y, epsilon, verbose, varargin{:});
                
                if verbose
                    fprintf('%d\t%f\n', it, train_nlZ(it));
                end
                
                % convergence
                converged = isConverged(train_nlZ, tol, 'objFun');
                if converged; break; end;
            end
        end

        function [nlZ, dnlZ] = gp_likelihood(m, Y, epsilon, varargin)
            assert(numel(varargin) >= 2, sprintf('%s needs region features for training', m.get_name()));
            X = varargin{2};

            n_var_args = length(varargin);
            if n_var_args == 3
                [m.hyp.lik, m.hyp.mean, m.hyp.cov, m.hyp.L] = m.get_hyperparameters(varargin{3});
            end

            [D, N] = size(Y);
            
            m.mu = feval(m.mean{:}, m.hyp.mean, X);
            m.Sigma = m.sn2*(feval(m.cov{:}, m.hyp.cov, X) + epsilon) + m.V*m.V';
            
            obs_sn2 = exp(2*m.hyp.lik);
            if obs_sn2 < 1e-6
                L_Sigma = chol(m.Sigma + obs_sn2*eye(D));
                sl = 1;
            else
                sl = obs_sn2;
                L_Sigma = chol(m.Sigma/obs_sn2 + eye(D));
            end
            inv_Sigma = solve_chol(L_Sigma, eye(D))/sl;
            logdet_Sigma = sum(log(diag(L_Sigma)));
            
            nlZ = 0;
            dnlZ = struct('lik', 0, 'mean', zeros(size(m.hyp.mean)), 'cov', zeros(m.covariance_gradient_size(), 1));
           
            d_mean = cell(numel(m.hyp.mean), 1);
            d_cov = cell(numel(dnlZ.cov), 1);
            
            if nargout >= 2
                for i = 1:numel(m.hyp.mean)
                    d_mean{i} = feval(m.mean{:}, m.hyp.mean, X, i);
                end
                for i = 1:numel(dnlZ.cov)
                    d_cov{i} = feval(m.cov{:}, m.hyp.cov, X, [], i);
                    dnlZ.cov(i) = m.sn2*trace(inv_Sigma*d_cov{i})*N/2;
                end
            end
            
            for n = 1:N
                y = Y(:, n);
                alpha = inv_Sigma * (y - m.mu);
                Q = inv_Sigma - alpha*alpha';

                nlZ = nlZ + (y - m.mu)'*alpha/2 + logdet_Sigma + D*log(2*pi*sl)/2;
                
                if nargout >= 2
                    dnlZ.lik = dnlZ.lik + obs_sn2*trace(Q);
                    for i = 1:numel(m.hyp.mean)
                        dnlZ.mean(i) = dnlZ.mean(i) - d_mean{i}'*alpha;
                    end
                    for i = 1:numel(dnlZ.cov)
                        dnlZ.cov(i) = dnlZ.cov(i) - m.sn2*alpha'*d_cov{i}*alpha/2;
                    end
                end
            end
            
            nlZ = nlZ / N;
            dnlZ.lik = dnlZ.lik / N;
            for i = 1:numel(m.hyp.mean)
                dnlZ.mean(i) = dnlZ.mean(i) / N;
            end
            for i = 1:numel(dnlZ.cov)
                dnlZ.cov(i) = dnlZ.cov(i) / N;
            end
        end
        
        function K_combined = get_covariace_matrix(m, varargin)
            assert(numel(varargin) >= 2, sprintf('%s needs region features for training', m.get_name()));
            X = varargin{2};
            K_combined = m.sn2*feval(m.cov{:}, m.hyp.cov, X) + m.V*m.V';
        end
        
        function [f, df] = objective_function(vals, m, Y, epsilon, varargin)
            hyp_ = m.vals_to_hyp(vals);
                        
            [f, gradient] = m.gp_likelihood(Y, epsilon, varargin{:}, hyp_);
            df = [gradient.lik; gradient.mean; gradient.cov];
        end
        
        function [hyp_opt, nll] = learn_gp(m, Y, epsilon, verbose, varargin)
            vals0 = m.hyp_to_vals(m.hyp);
            
            minFunc_options = struct('Display', verbose == 2, ...
                'Method', 'lbfgs', ...
                'LS', 2, ...
                'DerivativeCheck', 'Off', ...
                'MaxIter', 100, ...
                'MaxFunEvals', 10*100, ...
                'TolFun', 1e-3, ...
                'TolX', 1e-3);
            
            [vals_opt, nll] = minFunc(@objective_function, vals0, minFunc_options, m, Y, epsilon, varargin{:});
                        
            hyp_opt = m.vals_to_hyp(vals_opt);
        end
    
        function [y_hat_mu, y_hat_var, y_hat_cov] = predict(m, y, obs_idx, test_idx, varargin)
            assert(isequal(size(y), size(obs_idx)), ...
                'Mask and data should be the same size');
            assert(isequal(size(y), size(test_idx)), ...
                'Mask and data should be the same size');
            
            assert(numel(varargin) >= 2, sprintf('%s needs region and vote features for prediction', m.get_name()));
            X_v = varargin{1};
            X_m = varargin{2};
            
            if isempty(m.mu)
                m.mu = feval(m.mean{:}, m.hyp.mean, X_m);
                m.Sigma = m.sn2*feval(m.cov{:}, m.hyp.cov, X_m) + m.V*m.V';
            end

            N_obs = sum(obs_idx);
            obs_sn2 = exp(2*m.hyp.lik);
            if obs_sn2 < 1e-6
                L_Sigma_obs = chol(m.Sigma(obs_idx, obs_idx) + obs_sn2*eye(N_obs));
                sl = 1;
            else
                L_Sigma_obs = chol(m.Sigma(obs_idx, obs_idx)/obs_sn2 + eye(N_obs));
                sl = obs_sn2;
            end
            
            % remove contribution from vote features regression
            tX_v = [1 X_v];
            y_hat_vote_reg = m.Beta' * tX_v';
            
            alpha = solve_chol(L_Sigma_obs, y(obs_idx) - y_hat_vote_reg(obs_idx) - m.mu(obs_idx))/sl;
            y_hat_mu = y_hat_vote_reg(test_idx) + m.mu(test_idx) + m.Sigma(obs_idx, test_idx)'*alpha;
            
            if nargout >= 2
                v = solve_chol(L_Sigma_obs, m.Sigma(obs_idx, test_idx))/sl;
                y_hat_cov = m.Sigma(test_idx, test_idx) - m.Sigma(obs_idx, test_idx)'*v;
                y_hat_var = diag(y_hat_cov);
            end
        end
        
        function K = covariance_matrix(m, X_m)
            K = m.sn2*feval(m.cov{:}, m.hyp.cov, X_m) + m.V*m.V';
        end

    end
end
 
