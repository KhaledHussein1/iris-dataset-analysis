from sklearn import datasets
import numpy as np
from scipy import stats

iris = datasets.load_iris()
x= iris.data
y = iris.target

print('# of samples(rows):', len(x)) 
# print(x)
print('# of target labels:', len(y)) 
# print(y)

# Extract first column (feature: Sepal Length) of iris dataset
FirstCol = x[:,0]
# print(FirstCol)

print("\n-- Summary Statistics for Sepal Length --")

# mean: average of a data set
mean = np.mean(FirstCol)
print("Mean:", mean)
# problem: mean is sensitive to outliers

# trimmed mean: an estimate less sensitive to outliers
trimmedMean = stats.trim_mean(FirstCol, 0.1) # trims 10% from left-end and 10% from right-end after sorted
print("Trimmed Mean:", trimmedMean)

# median: middle of a set of values
median = np.median(FirstCol)
print("Median:", median)

# mode: value that has the highest frequency
mode = stats.mode(FirstCol)
print("Mode:", mode)

# frequency: # times each unique value appears in the array
values, counts = np.unique(FirstCol, return_counts=True)
print("Unique Values:", values)
print("Frequencies:", counts)

# range: measures the disperson (or spread) of a set of values
range = np.max(FirstCol) - np.min(FirstCol)
print("Range: ", range)

# variance - preferred as a measure of spread
var = np.var(FirstCol)
print("Variance: ", var)

# standard deviation: square root of the variance.
std = np.std(FirstCol)
print("Standard deviation: ", std)

# percentile: p-th percentile is a value (say xp) in the array such that p% of the values in that array are less than the value xp  
p = np.percentile(FirstCol, 50) # returns 50th percentile, which is also equal to median
print("50th Percentile:", p)

'''
# Covariance matrix
# Covariance indicates the level to which two variables (features) vary together.
# A 0 indicates that two features do not have a linear relationship
# In nmupy cov function, each row of the array represents a variable (feature), and each column a single observation (sample) of all those variables.
# The following examples shows two features, [0, 1, 2] and [2, 1, 0], which correlate perfectly, but in opposite directions

xx = np.array([[0, 2], [1, 1], [2, 0]]).T
print(xx)
print(np.cov(xx))

#correlation coefficient matrix, which is a normalized covariance matrix; we should percieve this one as better understandable
print(np.corrcoef(xx))
'''
print("\n-- Covariance Matrix for Sepal Length & Sepal Width --")
firstTwoFeatures = x[:, :2].T # use transpose so that each row is a feature
# print(firstTwoFeatures)
print(np.cov(firstTwoFeatures))

print("\n-- Correlation Coefficient Matrix for Sepal Length & Sepal Width --")
print(np.corrcoef(firstTwoFeatures))
