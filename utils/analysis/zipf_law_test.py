import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import ks_2samp, chisquare, zipfian

import json
from collections import Counter

'''
REFERENCES AND USEFUL LINKS:

Wikipedia:
https://en.wikipedia.org/wiki/Zipf's_law
https://en.wikipedia.org/wiki/Kolmogorov-Smirnov_test
https://en.wikipedia.org/wiki/Chi-squared_test

Paper:
https://arxiv.org/abs/0706.1062

Others:
http://bactra.org/weblog/491.html
https://stats.stackexchange.com/questions/264431/how-to-determine-if-zipfs-law-can-be-applied
https://stats.stackexchange.com/questions/6780/how-to-calculate-zipfs-law-coefficient-from-a-set-of-top-frequencies
'''

class ZipfEstimator:

    def __init__(self, estimation_method="mle"):

        if estimation_method not in ["mle","mse"]:
            raise Exception("Estimation method must be one between 'mle' and 'mse'")

        self.estimation_method=estimation_method

        self.s = None

        self.frequencies = None
        self.distribution = None
        self.expected_distribution = None
        self.expected_frequencies = None

        self.fitted = False

        self.ks_results = None
        self.chisquare_results = None

    def fit(self, frequencies):

        self.frequencies = np.array(frequencies)
        self.distribution = self._frequencies_to_probabilities(frequencies)

        objective_function = lambda s: self._mse_estimation(s) if self.estimation_method=="mse" else self._log_likelihood_estimation(s)
        minimization_results = minimize_scalar(objective_function, method="bounded", bounds=(0,100))

        self.s = minimization_results.x
        self.expected_distribution = self._zipf_distribution(self.s, len(self.frequencies))
        self.expected_frequencies = self._probabilities_to_frequencies(self.expected_distribution, np.sum(self.frequencies))

        self.fitted = True

    def goodness_of_fit(self):

        if not self.fitted:
            raise Exception("Estimator not fitted")

        self.ks_results = ks_2samp(self.distribution, self.expected_distribution)

        adjusted_expected_frequencies = np.copy(self.expected_frequencies)
        freq_diff = np.sum(self.frequencies) - np.sum(self.expected_frequencies)
        if freq_diff!=0:
            adjusted_expected_frequencies[0]+=freq_diff

        self.chisquare_results = chisquare(self.frequencies, f_exp=adjusted_expected_frequencies)

    def get_results(self):

        if not self.fitted:
            raise Exception("Estimator not fitted")

        results = {
        "s": self.s,
        "ks statistics": self.ks_results[0],
        "ks p-value": self.ks_results[1],
        "chisquare statistics": self.chisquare_results[0],
        "chisquare p-value": self.chisquare_results[1]
        }

        return results

    def compare_with_zipf(self, file_path=None):

        if not self.fitted:
            raise Exception("Estimator not fitted")

        for i, frequency in enumerate(self.frequencies):
            plt.bar(i, frequency, width=.5, color='blue')

        plt.plot(np.arange(len(self.expected_frequencies)), self.expected_frequencies, color="red", label="expected Zipf")

        plt.xlabel("clusters")
        plt.ylabel("frequencies")
        plt.legend()

        if file_path is None:
            plt.show()
        else:
            plt.savefig(file_path, bbox_inches='tight')
            plt.close(fig)

    def _generalized_harmonic_number(self, n, s):

        harmonic_number = 0
        for i in range(1,n+1):
            harmonic_number+= 1/(i**s)

        return harmonic_number

    def _zipf_log_distribution(self, s, max_rank):

        norm_term = self._generalized_harmonic_number(max_rank, s)

        log_zipf_law = lambda i, h, s: -s * math.log(i) - math.log(h)

        ranks = range(1,max_rank+1)
        zipf_log_distribution = [log_zipf_law(i, norm_term, s) for i in ranks]

        return np.array(zipf_log_distribution)

    def _zipf_distribution(self, s, max_rank):

        norm_term = self._generalized_harmonic_number(max_rank, s)

        zipf_law = lambda i, h, s: 1/(h * (i**s))

        ranks = range(1,max_rank+1)
        zipf_distribution = [zipf_law(i, norm_term, s) for i in ranks]

        return np.array(zipf_distribution)

    def _log_likelihood_estimation(self, s):

        max_rank = len(self.frequencies)

        norm_term = self._generalized_harmonic_number(max_rank, s)

        total_frequencies = np.sum(self.frequencies)

        ranks_likelihood = 0
        for i, f_i in enumerate(self.frequencies):
            ranks_likelihood+=f_i*math.log(i+1)

        likelihood = s * ranks_likelihood + total_frequencies * math.log(norm_term)

        return likelihood

    def _mse_estimation(self, s):

        max_rank = len(self.frequencies)

        real_log_probabilities = np.log(self.distribution)
        expected_log_probabilities = self._zipf_log_distribution(s, max_rank)

        mse = np.square(expected_log_probabilities - real_log_probabilities).mean()

        return mse

    def _frequencies_to_probabilities(self, frequencies):

        return frequencies/np.sum(frequencies)

    def _probabilities_to_frequencies(self, probabilities, total):

        return np.rint(probabilities*total)

#-------------------------------------------------------------------------------
#Test

def test():
    s, n = 1.25, 100
    x = np.arange(zipfian.ppf(0.01, s, n),
                  zipfian.ppf(0.99, s, n))
    frequencies = np.rint(10000*zipfian.pmf(x, s, n))

    zipf = ZipfEstimator(estimation_method="mle")
    zipf.fit(frequencies)
    zipf.goodness_of_fit()
    print(zipf.get_results())
    zipf.compare_with_zipf()

#-------------------------------------------------------------------------------
#Main

def zipf_estimation(json_path):

    #test()

    with open(json_path) as json_file:
        data = json.load(json_file)
        clusters_labels = data['labels']

        c = Counter(clusters_labels)
        frequencies = [element[1] for element in c.most_common()]

        zipf = ZipfEstimator(estimation_method="mle")
        zipf.fit(frequencies)
        zipf.goodness_of_fit()

        results = zipf.get_results()
        print(results)

        zipf.compare_with_zipf()

#zipf_estimation("NSCNet_kmeans_K128.json")
