function set_seed(seed)
% Manually set the seed of random number generators
    global RNDN_STATE  RND_STATE
    RNDN_STATE = randn('state');
    randn('state',seed);
    RND_STATE = rand('state');
    rand('twister',seed);
end