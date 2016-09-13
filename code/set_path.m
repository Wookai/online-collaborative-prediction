% Add necessary folders to the path

file_name = mfilename;
full_path = which(file_name);
current_dir = full_path(1:end-2-numel(file_name));

addpath(current_dir(1:end-1));
addpath([current_dir, 'gpml']);
addpath([current_dir, 'minFunc']);
addpath([current_dir, 'minFunc', '/compiled']);
addpath([current_dir, 'models']);
addpath([current_dir, 'utils']);

% GPML startup
startup();