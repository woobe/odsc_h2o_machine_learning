
# Start and connect to a local H2O cluster
suppressPackageStartupMessages(library(h2o))
h2o.init(nthreads = -1)

# Import Titanic data (local CSV)
titanic = h2o.importFile("kaggle_titanic.csv")

# Explore the dataset using various functions
head(titanic, 10)

# Explore the column 'Survived'
h2o.describe(titanic[, 'Survived'])

# Use hist() to create a histogram
h2o.hist(titanic[, 'Survived'])

# Use table() to summarize 0s and 1s
h2o.table(titanic[, 'Survived'])

# Convert 'Survived' to categorical variable
titanic[, 'Survived'] = as.factor(titanic[, 'Survived'])

# Look at the summary of 'Survived' again
# The feature is now an 'enum' (enum is the name of categorical variable in Java)
h2o.describe(titanic[, 'Survived'])

# Explore the column 'Pclass'
h2o.describe(titanic[,'Pclass'])

# Use hist() to create a histogram
h2o.hist(titanic[, 'Pclass'])

# Use table() to summarize 1s, 2s and 3s
h2o.table(titanic[, 'Pclass'])

# Convert 'Pclass' to categorical variable
titanic[, 'Pclass'] = as.factor(titanic[, 'Pclass'])

# Look at the summary of 'Pclass' again
# The feature is now an 'enum' (enum is the name of categorical variable in Java)
h2o.describe(titanic[, 'Pclass'])
