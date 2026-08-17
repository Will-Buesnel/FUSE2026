## Notes on Lab data:*
MLP001_wltp_25degC.xlsx: raw data from our battery cycler, where the current has been imposed and the voltage has been measured
wltp.csv: the current that was imposed. This is a drive-cycle file, giving current as a function of time. This plus the voltage data are useful for validating battery models.
MLP001_params.csv: resistance and timescale parameters for a cell, for various states of charge and temperatures
MLP001_ocv.csv: open-circuit voltage data for the same cell

*Lab data is currently gitignored.


Reproducing fig 11. :  
Fig. 11 compares model and experimental data on a drive-cycle with highly variable currents, representative of the power demands on an electric vehicle. Notably, the currents throughout the drive cycle are different to those used for parameterisation; this will increase the model error, but offers a useful test of how well the model performs on unseen data. A capacity of 2.13 Ah is used in the model, which reflects capacity losses in the cell after the parameterisation and before the validation experiments. The model generally performs well, and model errors compare favourably to literature results


# Assumptions
The electrical model is a Thevening-style ECM with two RC pairs as it has shown to create a better fit, even if it is more compuationally expensive.

Capacity is it's current state of charge.

Will assume Cough-Tocher interpolation is applicable to all interpolants, but this could be subject to further investigation later. it does by default cause NaN values to appear when the known bounds are exceeded, so will need to be aware of this.

# Current questions:
For the sensitivity analysis, how should interpolation error be considered?

Where does the thermal/temperature data come from?
    How do I validate the thermal data?

A point I am generally confused about is surely my investigation into how much error is in each parameter / each module of the model is just going to be some function of my assumptions about how much error there is originally? I'm not sure how the backprop factors in either.




# Code functionality
Basemodel.unpack(y) assumes the vector y is either made up by a flat list of scalars or will also work elementwise if arrays are passed in.
I used pack and unpack so that I didn't have to hardcode the amount of states in each model when I couple them up. it also makes pulling values from the state less prone to errors as I don't have to remember the indicie for a given state.

For parameter interpolation, all methods will default to clough-tocher, although I have kept funcionality to easily change this.


currently I don't have an interpolator for c1 and c2;  I just get the value from the interpolated values of r and tau





# Issues encountered with the dataset mlp001_params.csv.
 1. Soc datapoints seem to be taken at almost regular intervals for the most part, with slight differences. This is s.t if we rounded the soc points to 3dp, we would almost have a structured grid (which would make interpolation much easier/stable.)
    I am not sure whether this is simply a sensing resolution issue (which we could therefore ignore), or if this is something we cannot ignore/clean
    1. (b)  towards low soc values, the structure of the grid is inexact. Specifically, the final two points at the low temp of 5 degrees differs from the rest (which do hold the structured shape.)

2.  An issue related to the above is that because of these slight discrepancies, interpolation done through triangulation-based methods (e.g. Clough-Tocher) throws errors, because of tiny differences in points causing unstable gradient estimation. This propagates to every triangule sharing that vertex.
I suspect gradient esimation will be needed for the Bayesian estimation/inference required later on in the project.

3. 17 of the rows of the dataset have parameters which seem to be at some fixed bounds.
Specifically these are: 
R1 ≈ R2 ≈ 0.1 Ω
tau1 ≈ tau2 ≈ 180 s
All 17 of these rows fall within low SOC values (betw. 0 & 0.365), and across every sampled temperature.

# Issues with MPL_001_wlp_xxdeg_c_record excel file.
Due to a possible lack of resolution on time, there are multiple datapoints for the same time.
The temperature which it says it is at is not the actual temperature. e.g. for 25 deg, it is actually 23.9. It's not a massive issue, but it means that when calculating the errors we have to resample the experiment. This means that the errors we get will not be entirely accurate, with some portion being down to how we resample it.
I'm going to stick with resampling for now because that was what was done previously, but it something I will think about changing in the future. Another point with this is when we plot the errors in 'fig 11.', we're plotting the experiment data with errors from the resampled data so its inconsistent. I'm also thinking that maybe having multiple data points for the same time leads to unreliable sampling since it is again done via interpolation.
Doing it the above way also makes it much harder to manipulate if you only want to show the first x datapoints in a results graph.

will try de-quantisation instead of sampling.

# Points of note with equations
In Mark's code, there is a discrepancy between the equations given and what is in the code.
for soc_dot, he flips the polarity of the current, whereas in the equations he keeps it positive
A change of sign convention wouldn't normally be an issue, but in. the Theveninmodel class line 128, you can see that this change in sign convention is not kept. I think maybe the slides are wrong? In any case, I have made my code refect his, rather than the slides. 

# Amey suggestions:
.Put an event in the solver for when you reach the floor of soc, since this does not behave linearly and makes the error look unusually large.
.He agrees that trying dequantisation would be a useful addition.
.I need to decrease max step size to be smaller than the resolution of the experiment data.

# Notes on couped model:
assuming far field temperature == ambient temperature.
From more research, it seems that the models need to be differentiable in order for most Bayesian inference methods to work.

# Notes on adding Bayesian inference.
How do I get my prior distributions? resolution of meters?  
    Use the dataset to get distribution, i.e. empirical Bayes. Bootstrapping.

# Thermal model
should temperature be in Kelvin or degrees?

# Gaussian Process:

not sure if my current implementation meets the correct objective of 'only realising the process once per iteration'.
To meet this requirement, I still think we will need to interpolate again on the new function potentially?


# Downsampling plan

compare error at parameter-dataset given soc points, and then also at intepolated soc points. can use this to see if we need to decouple the error. 6, 2026


https://lacerbi.github.io/blog/2024/vi-is-inference-is-optimization/


# Using the Likelihood function as a variational parameter.

Is the model not going to want a likelihood function with more noise, because that means that the model results is more likely to be in the range of plausible values.
It kind of goes back to how does the MonteCarlo evaluate success.

i understand thought that having it as a varaitional parameter is good because it (could) show how much model error there is.. a least in a sense.