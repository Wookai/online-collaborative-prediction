# Online Collaborative Prediction of Regional Vote Results

This repository contains the source code and data used in the paper [Online Collaborative Prediction of Regional Vote Results](http://vincent.etter.io/pubications/etter2016dsaa.pdf), published in October 2016 by Vincent Etter, Mohammad Emtiyaz Khan, Matthias Grossglauser, and Patrick Thiran at the 3rd IEEE International Conference on Data Science and Advanced Analytics.

If you make use of the code or the data, please cite our paper:

```
@inproceedings{etter2016dsaa,
	author = "Etter, Vincent and Khan, Mohammad Emtiyaz and Grossglauser, Matthias and Thiran, Patrick",
	title = "Online Collaborative Prediction of Regional Vote Results",
	booktitle = "Proceedings of the 3rd IEEE International Conference on Data Science and Advanced Analytics",
	series = "DSAA '16",
	year = "2016",
}
```

To have more detailed information, please check the companion website: http://vincent.etter.io/dsaa16.

## Data

The dataset used in this paper has three data files: vote results, vote features, and region features. We give more details about each file below.

### Vote results

The first file `data/votes-data.mat` contains the outcome of 289 nationwide votes that took place in Switzerland between January 1981 and December 2014. For each vote, we record the result (proportion of yes) in every [Swiss municipalities](https://en.wikipedia.org/wiki/Municipalities_of_Switzerland) (municipalities are the smallest government division in Switzerland). The raw data can be obtained from the [Swiss Federal Statistical Office](http://www.bfs.admin.ch/bfs/portal/fr/index/themen/17/03/blank/data/01.html).

In December 2014, there were 2352 municipalities in Switzerland. However, because of administrative fusions and divisions, the regions change over time. To have a complete dataset over the 34 years spanned by the votes, we used a fixed set of regions (the ones exisiting in December 2014) and interpolated the result of regions that did not exist at some point in time. To compute the interpolated result of a region at a given time, we simply kept track all the fusions and divisions, took all regions existing at the time of the vote that have a part in the missing region, and computed the average of their results, weighted by their population. To have more details about the interpolation procedure, please refer to Section 5.2.1 of [this thesis](http://localhost:8000/publications/etter2015thesis.pdf).

The resulting vote results are stored as a `2352x289` matrix in `data/votes-data.mat`. For some votes, however, the interpolation procedure failed, e.g. when historical data was missing. When loading the data (see `code/utils/load_data.m`), we thus discard these votes, resulting in 281 votes for our analysis.

### Vote features

For each vote, the Swiss political parties emit voting recommendations, such as in favor, against, or no recommendation. These recommendations can be obtained from the [Swiss Federal Statistical Office](http://www.bfs.admin.ch/bfs/portal/fr/index/themen/17/03/blank/data/01.html).

We gathered recommendations from 25 parties, and stored them in the file data/votes-features.mat. It's a `289x26` matrix, where the first column contains the ID of the vote, and the rest of the columns the vote recommendations (encoded as `-1` for `against`, `0 ` for `no recommendation`, and `1` for `in favor`).

Some parties did not exist for all votes (in which case we encode it as 0), or always emit the same recommendations. For these reasons, we only keep parties (columns of the votes features matrix) that have a variance larger than 0.5 (see `code/utils/load_data.m`).

### Region features

Each region is characterized by 25 features: location, elevation, demographic attributes, political profile, and languages spoken. The data was obtained directly from the [Swiss Federal Statistical Office](http://www.bfs.admin.ch/bfs/portal/en/index/regionen/02/daten.html).

These features are stored as a `2352x25` matrix in the file `data/regions-features.mat`.

We summarize below the features and their main statistics. The x and y coordinates are defined in the Swiss coordinate system. The election features correspond to the proportion of votes for each of the main political parties during the national elections of 2011.

## Code

All models are implemented in `Matlab` (we used version `R2014b`). We make use of [GPML](http://www.gaussianprocess.org/gpml/code/matlab/doc/) and [minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html) for the Gaussian Process-based models.

To train all the models described in the paper, and then get the performance results on the test set, run `code/run.m`. It trains each model, saves the resulting model in the `models` folder, computes the prediciton accuracy on the test votes, and stores the results in the `results` folder.

To see the implementation of the models, look at the classes in the `code/models` folder. All models share a common interface, with the following methods:

```
% Fit the model m to the given training data Y, indicated by the indicator matrix train_idx.
% The training error (RMSE or negative loglikelihood) is returned, along with the validation
% RMSE computed on the entries indicated by valid_idx.
[train_error, valid_rmse] = fit(m, Y, train_idx, valid_idx, options, varargin);

% Predict the elements of y identified by test_idx, when observing
% the elements identified by obs_idx.
y_hat = predict(m, y, obs_idx, test_idx, varargin);
```

The models are named following the paper naming convention. For example, `model_mf_gp_r_liniso.m` corresponds to the `MF + GP(r) (linear)` model.

## Plots

The `plots` folder contains scripts to produce some of the figures of the papers. Once you have run `code/run.m`, you can use them to generate the plots.

## Contact

If you have any question about the code or the data, feel free to email vincent@etter.io.