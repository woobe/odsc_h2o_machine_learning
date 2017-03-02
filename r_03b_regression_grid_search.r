
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

# Build a Gradient Boosting Machines (GBM) model with default settings
gbm_default = h2o.gbm(x = features,
                      y = 'quality',
                      training_frame = wine_train,
                      seed = 1234,
                      model_id = 'gbm_default')

# Check the model performance on test dataset
h2o.performance(gbm_default, wine_test)

# Build a GBM with manual settings
gbm_manual = h2o.gbm(x = features,
                     y = 'quality',
                     training_frame = wine_train,
                     seed = 1234,
                     model_id = 'gbm_manual',
                     ntrees = 100,
                     sample_rate = 0.9,
                     col_sample_rate = 0.9)

# Check the model performance on test dataset
h2o.performance(gbm_manual, wine_test)

# Build a GBM with manual settings & cross-validation
gbm_manual_cv = h2o.gbm(x = features,
                        y = 'quality',
                        training_frame = wine_train,
                        seed = 1234,
                        model_id = 'gbm_manual_cv',
                        ntrees = 100,
                        sample_rate = 0.9,
                        col_sample_rate = 0.9,
                        nfolds = 5)

# Check the cross-validation model performance
gbm_manual_cv

# Check the model performance on test dataset
h2o.performance(gbm_manual_cv, wine_test)
# It should be the same as gbm_manual above as the model is trained with same parameters

# Build a GBM with manual settings, CV and early stopping
gbm_manual_cv_es = h2o.gbm(x = features,
                           y = 'quality',
                           training_frame = wine_train,
                           seed = 1234,
                           model_id = 'gbm_manual_cv_es',
                           ntrees = 10000,              # increase the number of trees
                           sample_rate = 0.9,
                           col_sample_rate = 0.9,
                           nfolds = 5,
                           stopping_metric = 'MSE',     # let early stopping feature determine
                           stopping_rounds = 15,        # the optimal number of trees
                           score_tree_interval = 1)     # by looking at the MSE metric

# Check the model summary
# which also includes cross-validation model performance
summary(gbm_manual_cv_es)

# Check the model performance on test dataset
h2o.performance(gbm_manual_cv_es, wine_test)

# define the criteria for full grid search
search_criteria = list(strategy = "Cartesian")

# define the range of hyper-parameters for grid search
param_list <- list(
  sample_rate = c(0.7, 0.8, 0.9),
  col_sample_rate = c(0.7, 0.8, 0.9)
)

# Set up GBM grid search
# Add a seed for reproducibility
# Full Grid Search
gbm_full_grid <- h2o.grid(
  
    # Core parameters for model training
    x = features,
    y = 'quality',
    training_frame = wine_train,
    ntrees = 10000,
    nfolds = 5,
    seed = 1234,

    # Parameters for grid search
    grid_id = "gbm_full_grid",
    hyper_params = param_list,
    algorithm = "gbm",
    search_criteria = search_criteria,

    # Parameters for early stopping
    stopping_metric = "MSE",
    stopping_rounds = 15,
    score_tree_interval = 1
  
)

# Sort and show the grid search results
gbm_full_grid <- h2o.getGrid(grid_id = "gbm_full_grid", sort_by = "mse")
print(gbm_full_grid)

# Extract the best model from full grid search
best_model_id <- gbm_full_grid@model_ids[[1]] # top of the list
best_gbm_from_full_grid <- h2o.getModel(best_model_id)
summary(best_gbm_from_full_grid)

# Check the model performance on test dataset
h2o.performance(best_gbm_from_full_grid, wine_test)

# define the criteria for random grid search
search_criteria = list(strategy = "RandomDiscrete",
                       max_models = 9,
                       seed = 1234)

# define the range of hyper-parameters for grid search
# 27 combinations in total
param_list <- list(
    sample_rate = c(0.7, 0.8, 0.9),
    col_sample_rate = c(0.7, 0.8, 0.9),
    max_depth = c(3, 5, 7)
)

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
    hyper_params = param_list,
    algorithm = "gbm",
    search_criteria = search_criteria,

    # Parameters for early stopping
    stopping_metric = "MSE",
    stopping_rounds = 15,
    score_tree_interval = 1
  
)

# Sort and show the grid search results
gbm_rand_grid <- h2o.getGrid(grid_id = "gbm_rand_grid", sort_by = "mse", decreasing = FALSE)
print(gbm_rand_grid)

# Extract the best model from random grid search
best_model_id <- gbm_rand_grid@model_ids[[1]] # top of the list
best_gbm_from_rand_grid <- h2o.getModel(best_model_id)
summary(best_gbm_from_rand_grid)

# Check the model performance on test dataset
h2o.performance(best_gbm_from_rand_grid, wine_test)

h2o.performance(best_gbm_from_rand_grid, wine_test)@metrics$MSE

cat('GBM with Default Settings                        :', 
          h2o.performance(gbm_default, wine_test)@metrics$MSE, "\n")
cat('GBM with Manual Settings                         :', 
          h2o.performance(gbm_manual, wine_test)@metrics$MSE, "\n")
cat('GBM with Manual Settings & CV                    :', 
          h2o.performance(gbm_manual_cv, wine_test)@metrics$MSE, "\n")
cat('GBM with Manual Settings, CV & Early Stopping    :', 
          h2o.performance(gbm_manual_cv_es, wine_test)@metrics$MSE, "\n")
cat('GBM with CV, Early Stopping & Full Grid Search   :', 
          h2o.performance(best_gbm_from_full_grid, wine_test)@metrics$MSE, "\n")
cat('GBM with CV, Early Stopping & Random Grid Search :', 
          h2o.performance(best_gbm_from_rand_grid, wine_test)@metrics$MSE, "\n")
