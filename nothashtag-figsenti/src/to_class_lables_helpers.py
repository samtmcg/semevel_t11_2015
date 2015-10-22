import numpy as np

def to_class_lables(score_list,number_of_classes):
	lables = ['A','B','C','D','E','F','G','H','I','J','K']
	matching_scores_to_lables = [[], [], [], [], [], [], [], [], [], [], []]
	rounded_score_list = [round(x) for x in score_list]
	class_labels=[]



	if number_of_classes == 11:
		all_possible_scores = range(-5,6)
		score_label_dict = dict()
		for i in range(number_of_classes):
			score_label_dict[all_possible_scores[i]] = lables[i]


		for i in range(len(score_list)):
			label = score_label_dict[rounded_score_list[i]]
			lab_index = lables.index(label)
			matching_scores_to_lables[lab_index].append(score_list[i])
			class_labels.append(label)
		categories = lables
	
	else:
		start       = 100/float(number_of_classes)
		edges       = [start*i for i in range(1,number_of_classes+1)]
		
		percentiles = np.percentile(score_list,edges)
		

		categories = lables[:number_of_classes]
		print 'PERCENTILES,:::,',percentiles

		for i in range(len(score_list)):
			score = rounded_score_list[i]
			actual_values_score = score_list[i]
			

			for a in range(number_of_classes):
				if a == 0:
					if score < percentiles[a]:
						label = lables[a]
						matching_scores_to_lables[a].append(actual_values_score)
						#print "score/label: ", str(score) +"/" + str(label)
				elif a >0 and a < number_of_classes-1:
					b = a-1
					if score >= percentiles[b] and score < percentiles[a]:
						label=lables[a]
						matching_scores_to_lables[a].append(actual_values_score)
						#print "score/label: ", str(score) +"/" + str(label)
				elif a == number_of_classes-1:
					b = a-1
					if score>= percentiles[b] and score <= percentiles[a]:
						label = lables[a]
						matching_scores_to_lables[a].append(actual_values_score)
						#print "score/label: ", str(score) +"/" + str(label)

			class_labels.append(label)
	return class_labels,categories,matching_scores_to_lables





def back_to_numbers(class_scores,scores_to_lables_lists,numclass):
	from scipy.stats import mode
	back_to_values_mean   = np.zeros(len(class_scores))
	back_to_values_mode_small   = np.zeros(len(class_scores))
	back_to_values_mode_larger   = np.zeros(len(class_scores))
	back_to_values_median = np.zeros(len(class_scores))
	back_to_values_max    = np.zeros(len(class_scores))
	back_to_values_min    = np.zeros(len(class_scores))
	lables = ['A','B','C','D','E','F','G','H','I','J','K']
	numbers_lables_dict = dict()
	for j in range(0,11):
		numbers_lables_dict[lables[j]] = j

	for i in range(len(class_scores)):
		cs = class_scores[i]
		bin = numbers_lables_dict[cs]

		back_to_values_mean[i]  		= np.array(scores_to_lables_lists[bin]).mean()
		back_to_values_mode_small[i]   	= mode(scores_to_lables_lists[bin])[0][0]
		back_to_values_mode_larger[i] 	= mode(scores_to_lables_lists[bin])[1][0] 
		back_to_values_median[i] 		= np.median(scores_to_lables_lists[bin])
		back_to_values_max[i]    		= np.array(scores_to_lables_lists[bin]).max()
		back_to_values_min[i]    		= np.array(scores_to_lables_lists[bin]).min()
		
		





	return [back_to_values_mean,back_to_values_mode_small,back_to_values_mode_larger,back_to_values_median,back_to_values_max,back_to_values_min ] 
