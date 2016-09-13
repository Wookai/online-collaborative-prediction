classdef model_mf_gp_r_seard < model_mf_gp_r   
    % MF + GP(r) model
    % The GP kernel is a squared-exponential ARD kernel
     
    methods        
        function m = model_mf_gp_r_seard(varargin)
            m = m@model_mf_gp_r(varargin{:});
        end
        
        function [mean, cov] = get_mean_cov(~)
            mean = {@meanZero};
            cov = {@covSEard};
        end
        
        function [lik_hyp, mean_hyp, cov_hyp, L] = get_hyperparameters(~, hyp)
            mean_hyp = []; % no params for zero mean
            %TODO fix hard-coded number of features
            [lik_hyp, cov_hyp, L] = myProcessOptions(hyp, 'lik', log(0.1), 'cov', randn(25 + 1, 1), 'L', 5);
            % make sure signal strenght is 1, as sn2 takes care of the
            % scale
            cov_hyp(end) = 0; % sf = 1 => log(sf) = 0
        end
        
        function name = get_name(~)
            name = 'MF + GP(r)';
        end
        
        function summary = get_params_summary(m)
            summary = sprintf('L=%d', m.hyp.L);
        end
        
        function suffix = get_filename_suffix(m)
            suffix = sprintf('mf_gp_r_seard_L=%d', m.hyp.L);
        end
        
        function vals = hyp_to_vals(~, hyp_)
            vals = [hyp_.lik; hyp_.mean; hyp_.cov(1:end-1)];
        end
        
        function hyp_ = vals_to_hyp(m, vals)
            n_hyp_lik = length(m.hyp.lik);
            n_hyp_mean = length(m.hyp.mean);
            
            hyp_.L = m.hyp.L;
            hyp_.lik = vals(1:n_hyp_lik);
            hyp_.mean = vals((1 + n_hyp_lik):(n_hyp_lik + n_hyp_mean));
            hyp_.cov = [vals((1 + n_hyp_lik + n_hyp_mean):end); 0]; % sf is always 1 => log(1) = 0
        end

        function d_cov_size = covariance_gradient_size(m)
            d_cov_size = length(m.hyp.cov) - 1;
        end
    end
end
 
