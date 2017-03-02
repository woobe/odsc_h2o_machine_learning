
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

# Build a Generalized Linear Model (GLM) with default settings
glm_default = h2o.glm(x = features,
                      y = 'quality',
                      training_frame = wine_train,
                      family = 'gaussian', 
                      model_id = 'glm_default')

# Check the model performance on training dataset
glm_default

# Check the model performance on test dataset
h2o.performance(glm_default, wine_test)

# Build a Distributed Random Forest (DRF) model with default settings
drf_default = h2o.randomForest(x = features,
                               y = 'quality',
                               training_frame = wine_train,
                               seed = 1234,
                               model_id = 'drf_default')

# Check the DRF model summary
drf_default

# Check the model performance on test dataset
h2o.performance(drf_default, wine_test)

# Build a Gradient Boosting Machines (GBM) model with default settings
gbm_default = h2o.gbm(x = features,
                      y = 'quality',
                      training_frame = wine_train,
                      seed = 1234,
                      model_id = 'gbm_default')

# Check the GBM model summary
gbm_default

# Check the model performance on test dataset
h2o.performance(gbm_default, wine_test)

# Build a Deep Learning (Deep Neural Networks, DNN) model with default settings
dnn_default = h2o.deeplearning(x = features,
                               y = 'quality',
                               training_frame = wine_train,
                               model_id = 'dnn_default')

# Check the DNN model summary
dnn_default

# Check the model performance on test dataset
h2o.performance(dnn_default, wine_test)

# Use GLM model to make predictions
yhat_test_glm = h2o.predict(glm_default, wine_test)
head(yhat_test_glm)

# Use DRF model to make predictions
yhat_test_drf = h2o.predict(drf_default, wine_test)
head(yhat_test_drf)

# Use GBM model to make predictions
yhat_test_gbm = h2o.predict(gbm_default, wine_test)
head(yhat_test_gbm)

# Use DNN model to make predictions
yhat_test_dnn = h2o.predict(dnn_default, wine_test)
head(yhat_test_dnn)
