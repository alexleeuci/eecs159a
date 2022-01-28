#do a chi-squared analysis on the distribution of 1 vs 0 values
import numpy as np
distr1 = np.array([1,2,3,4,5])
distr2 = np.array([15,4,3,2,1])
table = [distr1,distr2]
from scipy.stats import chi2_contingency
from scipy.stats import chi2
stat, p, dof, expected = chi2_contingency(table)
print("-----test vals-----")
print(stat)
print(p)
print(dof)
print(expected)
print("-----")
# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
