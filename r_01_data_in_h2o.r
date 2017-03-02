
# Start and connect to a local H2O cluster
suppressPackageStartupMessages(library(h2o))
h2o.init(nthreads = -1)

# Method 1 - Import data from a local CSV file
data_from_csv = h2o.importFile("winequality-white.csv")
head(data_from_csv, 5)

# Method 2 - Import data from the web
data_from_web = h2o.importFile("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv")
head(data_from_web, 5)

# Method 3 - Convert R data frame into H2O data frame

## Import Wine Quality data using R
wine_df = read.csv('winequality-white.csv', sep = ';')
head(wine_df, 5)

## Convert R data frame into H2O data frame
data_from_df = as.h2o(wine_df)
head(data_from_df, 5)
