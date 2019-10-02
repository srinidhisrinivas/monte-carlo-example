import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import pandas as pd
import threading

#creates a frozen distribution with a set of supplied parameters
class Distribution:
	dist = None;
	distclass = None;
	params = [];
	dist_map = {
		'Normal':stats.norm,
		'Uniform': stats.uniform,
		'Beta': stats.beta,
		'Bernoulli': stats.bernoulli
	}	
	def __init__(self, distname='Normal', params=[0, 1]):
		self.params = params;
		self.distclass = self.dist_map.get(distname, 'Invalid Distribution').__class__;
		self.dist = self.dist_map.get(distname, 'Invalid Distribution')(*self.params);

	def pdf_pmf(self, x):
		return self.dist.pdf(x) if issubclass(self.distclass, stats.rv_continuous) else self.dist.pmf(x);

	def rvs(self, size=1):
		return self.dist.rvs(size);

#computes the likelihood for a given data observation, distribution type, and parameter
def get_likelihood(data, dist_type, theta):
	l = 1;
	for index, row in data.iterrows():
		dist = Distribution(data_dist_type, theta);
		prob = dist.pdf_pmf(row['obs']);
		l = l*prob;
	return l;

#runs the metropolis hastings Markov Chain Monte Carlo algorithm to estimate the posterior distribution
#using a normal proposal distribution of fixed variance and initial mean (theta).
def metropolis_hastings(iterations, data, data_dist_type, theta_prior_dist, theta, sigma):
	guessed = [];
	accepted = [];
	prev_posterior = theta_prior_dist.pdf_pmf(theta) * get_likelihood(data, data_dist_type, [theta]);
	proposal_dist = Distribution('Normal', [theta, sigma]);

	for i in range(iterations):
		theta_guess = proposal_dist.rvs(1)[0];
		guessed.append(theta_guess);

		#compute new posterior value based on prior and guessed theta value
		new_posterior = theta_prior_dist.pdf_pmf(theta_guess) * get_likelihood(data, data_dist_type, [theta_guess]);
		post_ratio = new_posterior/prev_posterior;
		reject_threshold = min(post_ratio, 1);
		reject_check = random.random();
		if reject_check < reject_threshold:
			theta = theta_guess;
			accepted.append(theta);
			prev_posterior = new_posterior;
			proposal_dist = Distribution('Normal', [theta, sigma]);


	print('Metropolis-Hastings', len(accepted), len(guessed))

	plt.figure(1);
	plt.clf();
	plt.hist(accepted, bins=50, density=True);
	plt.title('Posterior Distribution w/ Metropolis Hastings');
	xmin, xmax = plt.xlim();
	x = np.linspace(xmin, xmax, 100);
	plt.plot(x, theta_prior_dist.pdf_pmf(x), 'k', linewidth=2);

#runs a simulated annealing schedule to find the optimal parameter based on maximal posterior probability 
def simulated_annealing(iterations, data, data_dist_type, theta_prior_dist, theta, sigma):
	
	guessed = [];
	accepted = [];
	prev_posterior = theta_prior_dist.pdf_pmf(theta) * get_likelihood(data, data_dist_type, [theta]);
	proposal_dist = Distribution('Normal', [theta, sigma]);

	for i in range(iterations):
		#print('SA', i);
		theta_guess = proposal_dist.rvs(1)[0];
		guessed.append(theta_guess);
		new_posterior = theta_prior_dist.pdf_pmf(theta_guess) * get_likelihood(data, data_dist_type, [theta_guess]);
		
		#compute change in energy from previous state
		d_E = new_posterior - prev_posterior;

		#create annealing schedule
		T = float(iterations - i) * 10**(-10);
		
		if (d_E > 0) or (math.exp(d_E/T) > random.random()):
			theta = theta_guess;
			accepted.append(theta);
			prev_posterior = new_posterior;
			#proposal_dist = Distribution('Normal', [theta, sigma]);

	print('Simulated Annealing', len(accepted), len(guessed));
	plt.figure(2);
	plt.clf();
	plt.hist(accepted, bins=50, density=True);
	plt.title('Posterior Distribution w/ Simulated Annealing');
	xmin, xmax = plt.xlim();
	x = np.linspace(xmin, xmax, 100);
	plt.plot(x, theta_prior_dist.pdf_pmf(x), 'k', linewidth=2);

if __name__ == '__main__':
	data = pd.read_csv('coin_flip_data.csv');
	theta_prior_dist = Distribution('Normal', [0.1, 0.4]);
	data_dist_type = 'Bernoulli';
	iterations = 5000;
	theta = [0.4, 0.5];
	sigma = [0.2, 0.5];

	#metropolis_hastings(iterations, data, data_dist_type, theta_prior_dist, theta[0], sigma[0]);
	simulated_annealing(iterations, data, data_dist_type, theta_prior_dist, theta[1], sigma[1]);

	plt.show();
