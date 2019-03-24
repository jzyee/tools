import matplotlib.pyplot as plt


########################################
#			randomness
########################################

def seeded_execution(filename, rounds=1):
	for x in range(rounds):
		var_seeding()
		exec(open(filename, 'r').read())

def var_seeding(seed=1):
	import numpy as np
	# Setting the seed for numpy-generated random numbers
	np.random.seed(seed)

	import random as rn
	# Setting the seed for python random numbers
	rn.seed(seed)

	import tensorflow as tf
	# Setting the graph-level random seed.
	tf.set_random_seed(seed)


	# Setting the number of cores used to 1
	session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
	                             inter_op_parallelism_threads=1)

	from keras import backend as K
	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
	K.set_session(sess)

########################################
#			visualization
########################################

def class_bin_diagrams(df, label, figsize):
    '''
    df           - pandas dataframe
    label        - str
    size_of_cell - tuple of ints
    '''    
    classes = len(df.groupby(label))
    fig, axes = plt.subplots(nrows=len(df.columns), ncols=classes)

    for idx, col in enumerate(df.columns):
        for x in range(classes):
            df[col].loc[df[label] == x].hist(figsize=figsize,ax=axes[idx,x])
    plt.plot()