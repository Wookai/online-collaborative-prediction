function [Y, t, X_U, X_V, population, global_id, X_U_raw] = load_data()
    % region results
    S = load('../data/votes-data.mat', 'Y');
    Y = S.Y;
    
    global_id = 1:size(Y, 2);
    
    % remove missing value votes
    valid_votes = sum(isnan(Y)) == 0;
    Y = Y(:, valid_votes);
    global_id = global_id(valid_votes);
    
    % normalize result
    Y = Y./100;
    
    if nargout >= 2
        % national results
        S = load('../data/national-results.mat', 't');
        t = S.t;
        
        t = t./100;
        t = t(valid_votes);
     
        if nargout >= 3
            % vote features
            S = load('../data/votes-features.mat', 'X_votes');
            X_U = S.X_votes;
            
            X_U = X_U(valid_votes, :);
            
            % only keep vote recommendations that have some signal
            useful_recom = var(X_U) > .5;
            X_U = X_U(:, useful_recom);
            if nargout >= 7
                X_U_raw = X_U;
                X_U_raw = bsxfun(@minus, X_U_raw, mean(X_U_raw));
                X_U_raw = bsxfun(@times, X_U_raw, 1./std(X_U_raw));
            end
            
            % add a tiny amount of noise to make all votes different
            X_U = X_U + randn(size(X_U))/10;
            
            % normalize features
            X_U = bsxfun(@minus, X_U, mean(X_U));
            X_U = bsxfun(@times, X_U, 1./std(X_U));
            
            if nargout >= 4
                % region features
                S = load('../data/regions-features.mat', 'X');
                X_V = S.X;

                % fill missing values in features with the average
                for i = 1:size(X_V,2)
                  missing_vals = isnan(X_V(:, i));
                  mean_ = mean(X_V(~missing_vals, i));
                  X_V(missing_vals, i) = mean_;
                end
                
                if nargout >= 5
                    % extract population before normalizing
                    population = X_V(:, 19);
                end

                % normalize features
                X_V = bsxfun(@minus, X_V, mean(X_V));
                X_V = bsxfun(@times, X_V, 1./std(X_V));
            end
        end
    end
end