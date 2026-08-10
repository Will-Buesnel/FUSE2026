"""
Will Buesnel, Aug 26
The file is used to look at the difference (if any) between the errors incurred by the deterministic model for v_cell at known soc points (i.e. ones in the parameter dataset)
and ones in between the known soc points (i.e. ones that are interpolated by the parameter interpolants). 
The errors are calculated by resampling the model results at the soc points in the parameter dataset and comparing them to the experimental data at those soc points.
The errors are then plotted as a function of soc.
First, we have to remove transient effects. Therefore, we can only take the errors at soc points on the flat parts of the results.
"""

