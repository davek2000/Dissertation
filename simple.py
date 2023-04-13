from __future__ import print_function

# For number crunching
import numpy as np
import pandas as pd

# For visualisation
import matplotlib.pyplot as pl 
import seaborn as sns 

# For prediction 
import sklearn

# Misc
from itertools import cycle
import json 
import os


sns.set_context('poster')
sns.set_style('darkgrid')

current_palette = cycle(sns.color_palette())

import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
    
public_data_path = 'data/'
metadata_path = 'data/metadata'
    
from visualise_data import SequenceVisualisation

plotter = SequenceVisualisation(metadata_path, public_data_path + '/train/00001')
sequence_window = (plotter.meta['start'], plotter.meta['end'])

# We can extract the time range of the activity 'jump' with the 'times_of_activity' function. 
# This function returns all of the times that jump was annotated, and it is indexed first 
# by the annotator, and then by the time at which it occurred. 
times_of_jump = plotter.times_of_activity('a_jump')

# You can get the times at which these. 
for ai, annotator_jumps in enumerate(times_of_jump): 
    print ('Annotator {}'.format(ai))
    
    # The annotator_jumps is a list of tuples. The length of this list specifies the number of segments as
    # annotated as 'a_jump' by this annotator. Each element of this list is a tuple that holds the start and 
    # end time of the t-th annotation in that order, ie: 
    for ti, (start, end) in enumerate(annotator_jumps, start=1): 
        print ('  Annotation {}'.format(ti))
        print ('    Start time: {}'.format(start))
        print ('    End time:   {}'.format(end))
        print ('    Duration:   {}'.format(end - start))
        print ()

# The sequence object also holds metadata regarding the length of the sequence. 
#sequence_window = (plotter.meta['start'], plotter.meta['end'])
#print (sequence_window)
#plotter.plot_pir(sequence_window, sharey=True)
#pl.show()

annotation_names = plotter.targets.columns.difference(['start', 'end'])

# Select only the first minute of data 
sub_df = plotter.targets.iloc[:60 * 2]

# Select only the columns that are non-empty 
sub_df_cols = [col for col in annotation_names if sub_df[col].sum() > 0]

# Plot a bar-plot w
current_palette = cycle(sns.color_palette())
sub_df[sub_df_cols].plot(
    kind='bar', 
    subplots=True, 
    sharex=True, 
    sharey=True, 
    figsize=(20, 3 * len(sub_df_cols)), 
    width=1.0, 
    color=[next(current_palette) for _ in annotation_names]
);
pl.show()
