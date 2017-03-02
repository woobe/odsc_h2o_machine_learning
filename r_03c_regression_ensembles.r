
# Start and connect to a local H2O cluster
suppressPackageStartupMessages(library(h2o))
h2o.init(nthreads = -1)

# Import wine quality data from a local CSV file
wine = h2o.importFile("winequality-white.csv")
head(wine, 5)

# Define features (or predictors)
features = colnames(wine)  # we want to use all the information
features = setdiff(features, 'quality')    # we need to exclude the target 'quality'
features

# Split the H2O data frame into training/test sets
# so we can evaluate out-of-bag performance
wine_split = h2o.splitFrame(wine, ratios = 0.8, seed = 1234)

wine_train = wine_split[[1]] # using 80% for training
wine_test = wine_split[[2]]  # using the rest 20% for out-of-bag evaluation

dim(wine_train)

dim(wine_test)

# define the criteria for random grid search
search_criteria = list(strategy = "RandomDiscrete",
                       max_models = 9,
                       seed = 1234)

# define the range of hyper-parameters for GBM grid search
# 27 combinations in total
hyper_params <- list(
    sample_rate = c(0.7, 0.8, 0.9),
    col_sample_rate = c(0.7, 0.8, 0.9),
    max_depth = c(3, 5, 7)
)

# Set up GBM grid search
# Add a seed for reproducibility
# Set up GBM grid search
# Add a seed for reproducibility
gbm_rand_grid <- h2o.grid(
  
    # Core parameters for model training
    x = features,
    y = 'quality',
    training_frame = wine_train,
    ntrees = 10000,
    nfolds = 5,
    seed = 1234,

    # Parameters for grid search
    grid_id = "gbm_rand_grid",
    hyper_params = hyper_params,
    algorithm = "gbm",
    search_criteria = search_criteria,

    # Parameters for early stopping
    stopping_metric = "MSE",
    stopping_rounds = 15,
    score_tree_interval = 1,
    
    # Parameters required for stacked ensembles
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = TRUE
  
)

# Sort and show the grid search results
gbm_rand_grid <- h2o.getGrid(grid_id = "gbm_rand_grid", sort_by = "mse", decreasing = FALSE)
print(gbm_rand_grid)

# Extract the best model from random grid search
best_gbm_model_id <- gbm_rand_grid@model_ids[[1]] # top of the list
best_gbm_from_rand_grid <- h2o.getModel(best_gbm_model_id)
summary(best_gbm_from_rand_grid)

# define the range of hyper-parameters for DRF grid search
# 27 combinations in total
hyper_params <- list(
    sample_rate = c(0.5, 0.6, 0.7),
    col_sample_rate_per_tree = c(0.7, 0.8, 0.9),
    max_depth = c(3, 5, 7)
)

# Set up DRF grid search
# Add a seed for reproducibility
drf_rand_grid <- h2o.grid(
  
    # Core parameters for model training
    x = features,
    y = 'quality',
    training_frame = wine_train,
    ntrees = 200,
    nfolds = 5,
    seed = 1234,

    # Parameters for grid search
    grid_id = "drf_rand_grid",
    hyper_params = hyper_params,
    algorithm = "randomForest",
    search_criteria = search_criteria,
    
    # Parameters required for stacked ensembles
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = TRUE
  
)

# Sort and show the grid search results
drf_rand_grid <- h2o.getGrid(grid_id = "drf_rand_grid", sort_by = "mse", decreasing = FALSE)
print(drf_rand_grid)

# Extract the best model from random grid search
best_drf_model_id <- drf_rand_grid@model_ids[[1]] # top of the list
best_drf_from_rand_grid <- h2o.getModel(best_drf_model_id)
summary(best_drf_from_rand_grid)

# define the range of hyper-parameters for DNN grid search
# 81 combinations in total
hyper_params <- list(
    activation = c('tanh', 'rectifier', 'maxout'),
    hidden = list(c(50), c(50,50), c(50,50,50)),
    l1 = c(0, 1e-3, 1e-5),
    l2 = c(0, 1e-3, 1e-5)
)

# Set up DNN grid search
# Add a seed for reproducibility
dnn_rand_grid <- h2o.grid(
  
    # Core parameters for model training
    x = features,
    y = 'quality',
    training_frame = wine_train,
    epochs = 20,
    nfolds = 5,
    seed = 1234,

    # Parameters for grid search
    grid_id = "dnn_rand_grid",
    hyper_params = hyper_params,
    algorithm = "deeplearning",
    search_criteria = search_criteria,
    
    # Parameters required for stacked ensembles
    fold_assignment = "Modulo",
    keep_cross_validation_predictions = TRUE
  
)

# Sort and show the grid search results
dnn_rand_grid <- h2o.getGrid(grid_id = "dnn_rand_grid", sort_by = "mse", decreasing = FALSE)
print(dnn_rand_grid)

# Extract the best model from random grid search
best_dnn_model_id <- dnn_rand_grid@model_ids[[1]] # top of the list
best_dnn_from_rand_grid <- h2o.getModel(best_dnn_model_id)
summary(best_dnn_from_rand_grid)

# Define a list of models to be stacked
# i.e. best model from each grid
all_ids = list(best_gbm_model_id, best_drf_model_id, best_dnn_model_id)

# Stack models
# GLM as the default metalearner
ensemble = h2o.stackedEnsemble(x = features,
                               y = 'quality',
                               training_frame = wine_train,
                               model_id = "my_ensemble",
                               base_models = all_ids)

cat('Best GBM model from Grid (MSE) : ', h2o.performance(best_gbm_from_rand_grid, wine_test)@metrics$MSE, "\n")
cat('Best DRF model from Grid (MSE) : ', h2o.performance(best_drf_from_rand_grid, wine_test)@metrics$MSE, "\n")
cat('Best DNN model from Grid (MSE) : ', h2o.performance(best_dnn_from_rand_grid, wine_test)@metrics$MSE, "\n")
cat('Stacked Ensembles        (MSE) : ', h2o.performance(ensemble, wine_test)@metrics$MSE, "\n")
