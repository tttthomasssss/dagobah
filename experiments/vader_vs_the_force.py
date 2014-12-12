from __future__ import division

import collections
import os
import sys

from matplotlib import pyplot as plt
from matplotlib import pylab as pl
import numpy as np
import seaborn as sns
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment


def _evaluate_vader(data, test_idx):
	'''
	Vader outputs a dictionary of sentiment probabilities, i.e.:

		In [9]: vs = vaderSentiment('This is utter crap!')
		In [10]: vs
		{'neg': 0.491, 'neu': 0.509, 'pos': 0.0, 'compound': -0.4389}

	--> I simply take the argmax over the neg, neu and pos probabilities as its prediction (I don't know and don't care what the compund is for)
	'''
	vader_prediction = []
	for idx in test_idx:
		vs = vaderSentiment(data[idx])
		vader_prediction.append(np.argmax(vs.values()[:3]))

	return vader_prediction


def _print_results(scores, score_errs, metric, out_folder):
	sns.set_palette('deep', desat=.6)
	sns.set_context(rc={'figure.figsize': (8, 4)})

	ymin, ymax = 0., 1.

	for a in scores.iterkeys():
		plot_path = os.path.join(out_folder, 'alpha_%.1f_%s.png' % (a, metric))

		idx = np.arange(len(scores[a].keys()))

		fig = pl.figure()
		plt.ylim(ymin, ymax)
		plt.xlabel('Classifier')
		plt.ylabel(metric)
		plt.title('%s for alpha=%f' % (metric, a))
		plt.xticks(idx, scores[a].keys())
		plt.grid(True)
		plt.hold(True)

		plt.errorbar(x=idx, y=scores[a].values(), yerr=score_errs[a].values())

		fig.savefig(plot_path)

		plt.close(fig)

def run(runs=10, folds=10, alpha=np.arange(0., 2.01, 0.1), in_file='/Volumes/LocalDataHD/thk22/Downloads/hutto_ICWSM_2014/tweets_GroundTruth.txt', out_folder='/Volumes/LocalDataHD/thk22/Downloads/hutto_ICWSM_2014/'):
	classifiers = {
		'linear_svm': SVC(kernel='linear'),
		'rbf_svm': SVC(kernel='rbf'),
		'mnb': MultinomialNB(alpha=0.001),
		'lr': LogisticRegression()
	}

	labels_per_alpha = collections.defaultdict(list)
	data = []

	# Read in stuff
	with open(in_file) as f:
		for line in f:
			parts = line.split('\t')
			score = float(parts[1])

			for a in alpha:
				if (score > a): # Following the key indices that vader returns
					labels_per_alpha[a].append(2) # Positive
				elif (score < -a):
					labels_per_alpha[a].append(0) # Negative
				else:
					labels_per_alpha[a].append(1) # Neutral

			data.append(parts[2])

	vec = CountVectorizer(decode_error='ignore')
	X = vec.fit_transform(data)

	random_state = np.random.RandomState(seed=1711)

	accs = collections.defaultdict(lambda: collections.defaultdict(list))
	accs_avg = collections.defaultdict(lambda: collections.defaultdict(list))
	accs_std = collections.defaultdict(lambda: collections.defaultdict(list))
	f1ss = collections.defaultdict(lambda: collections.defaultdict(list))
	f1ss_avg = collections.defaultdict(lambda: collections.defaultdict(list))
	f1ss_std = collections.defaultdict(lambda: collections.defaultdict(list))

	# Run classifiers
	for a, labels in labels_per_alpha.iteritems():
		#print 'CURRENT ALPHA=%f' % (a,)

		y = np.array(labels)

		# Label Distribution
		pos_ratio = np.where(y==2)[0].shape[0] / y.shape[0]
		neg_ratio = np.where(y==0)[0].shape[0] / y.shape[0]
		neu_ratio = np.where(y==1)[0].shape[0] / y.shape[0]
		print 'LABEL DISTRIBUTION'
		print '>    [ALPHA=%.1f]POS=%.2f; NEG=%.2f; NEU=%.2f' % (a, pos_ratio,  neg_ratio, neu_ratio)
		print '---------------------------------------------------------'

		try:
			for i in xrange(runs):
				print 'RUN: %d' % (i,)
				kfold = StratifiedKFold(y, folds, shuffle=True, random_state=random_state)
				for train_idx, test_idx in kfold:
					for key, cls in classifiers.iteritems():
						#print 'CLASSIFIER: %s' % (key,)
						X_train, y_train = X[train_idx], y[train_idx]
						X_test, y_test = X[test_idx], y[test_idx]

						cls.fit(X_train, y_train)
						prediction = cls.predict(X_test)

						accs[a][key].append(accuracy_score(y_test, prediction))
						f1ss[a][key].append(f1_score(y_test, prediction, pos_label=2, average='macro'))

					# Evaluate vader
					accs[a]['vader'] = accuracy_score(y_test, _evaluate_vader(data, test_idx))
					f1ss[a]['vader'] = f1_score(y_test, _evaluate_vader(data, test_idx))

			# Mean + Std Dev
			for key in classifiers.keys() + ['vader']:
				accs_avg[a][key] = np.average(accs[a][key])
				accs_std[a][key] = np.std(accs[a][key])
				f1ss_avg[a][key] = np.average(f1ss[a][key])
				f1ss_std[a][key] = np.std(f1ss[a][key])
		except ValueError, e:
			print '[FAIL] (but continue):', e


	_print_results(accs_avg, accs_std, 'Accuracy', out_folder)
	_print_results(f1ss_avg, f1ss_std, 'F1-Score', out_folder)

	print 'AVG:', accs_avg
	print 'F1:', f1ss_avg

if (__name__ == '__main__'):
	runs = sys.argv[1] if len(sys.argv) > 1 else 10
	in_file = sys.argv[2] if len(sys.argv) > 2 else '/Volumes/LocalDataHD/thk22/Downloads/hutto_ICWSM_2014/tweets_GroundTruth.txt'
	folds = sys.argv[3] if len(sys.argv) > 3 else 10
	out_folder = sys.argv[4] if len(sys.argv) > 4 else '/Volumes/LocalDataHD/thk22/Downloads/hutto_ICWSM_2014/'
	alpha = np.arange(2.1, 3.91, 0.1)

	run(runs=runs, folds=folds, alpha=alpha, in_file=in_file)