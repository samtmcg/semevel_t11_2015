from __future__ import print_function
import argparse
import sys
import numpy as np
from scipy import stats

parser = argparse.ArgumentParser(description="""Internal evaluation scripts""")
parser.add_argument('sys1')
parser.add_argument('sys2')
args = parser.parse_args()


def stderrout(*objs):
    print( *objs, file=sys.stderr)


def getScores(file):
    scores = []
    for line in open(file).readlines():
        a= line.strip().split()
        scores.append(float(a[1]))
    return np.array(scores)

sys1_scores = getScores(args.sys1)
sys2_scores = getScores(args.sys2)


r = stats.pearsonr(sys1_scores,sys2_scores)[0]

print(r)