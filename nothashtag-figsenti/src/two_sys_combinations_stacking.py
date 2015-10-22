import itertools
import sys
import os
import argparse
import subprocess


parser = argparse.ArgumentParser(description="""finding combinations of features.""")
parser.add_argument('scores_folder')
args = parser.parse_args()

scores_folder          			= args.scores_folder

all_model_output 				= os.listdir(scores_folder)
all_model_outputs = [x for x in all_model_output if not x == '.DS_Store']
number_outputs = len(all_model_output)


tmp_path = "../data/zvals/tmp/"
pf = subprocess.Popen(['rm', '-r',tmp_path])
outf,errf = pf.communicate()

training_parts     = [x for x in all_model_outputs if  'train.' in x]
testing_parts      = [y for y in all_model_outputs  if (('trial.' in y) or ('test.' in y))]

training_parts.sort()
testing_parts.sort()

### make all possible feature combinations
for L in range(2,3):

	training_combinations = itertools.combinations(training_parts, L)
	testing_combinations  = itertools.combinations(testing_parts,  L)

	for x,y  in zip(training_combinations,testing_combinations):
		subset = x+y


		# Path to be created
		tmp_path = '../data/zvals/two_tmp/'
		os.mkdir(tmp_path)

		linearregr_stacking_commands        = ['python','stacking.py','../data/zvals/two_tmp/','../data/twokenized/train.dat','-testScore','../data/twokenized/trial.dat','-model','linearregr']
		randomforrestregr_stacking_commands = ['python','stacking.py','../data/zvals/two_tmp/','../data/twokenized/train.dat','-testScore','../data/twokenized/trial.dat','-model','randomforrestregr']
		bayesianridge_stacking_commands     = ['python','stacking.py','../data/zvals/two_tmp/','../data/twokenized/train.dat','-testScore','../data/twokenized/trial.dat','-model', 'bayesianridge']
		ridgecv_stacking_commands           = ['python','stacking.py','../data/zvals/two_tmp/','../data/twokenized/train.dat','-testScore','../data/twokenized/trial.dat','-model', 'ridgecv']

		

		for sfile in subset:
			source = scores_folder + sfile
			p1 = subprocess.Popen(['cp', source,tmp_path])
			out1,err1 = p1.communicate()

		linearregr_proc = subprocess.Popen(linearregr_stacking_commands, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		linearregr_proc_out,linearregr_proc_err = linearregr_proc.communicate()

		randomforrestregr_stacking_commands_proc = subprocess.Popen(randomforrestregr_stacking_commands, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		randomforrestregr_stacking_commands_proc_out,randomforrestregr_stacking_commands_proc_err = randomforrestregr_stacking_commands_proc.communicate()


		bayesianridge_stacking_commands_proc = subprocess.Popen(bayesianridge_stacking_commands, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		bayesianridge_stacking_commands_out,linearregr_proc_err = bayesianridge_stacking_commands_proc.communicate()

		ridgecv_stacking_commands_proc = subprocess.Popen(ridgecv_stacking_commands, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		ridgecv_stacking_commands_proc_out,linearregr_proc_err = ridgecv_stacking_commands_proc.communicate()



		combinations = ('+').join(subset)

		print combinations +',' + 'linearregr'+',' + linearregr_proc_out
		print combinations+',' +'randforrest' +',' + randomforrestregr_stacking_commands_proc_out
		print combinations +',' + 'baysridge'+',' + bayesianridge_stacking_commands_out
		print combinations +',' + 'ridgecv' + ',' + ridgecv_stacking_commands_proc_out


		pf = subprocess.Popen(['rm', '-r',tmp_path])
		outf,errf = pf.communicate()

            

