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
Should capacity in the soc ode be defined as instantaneous or maximal capacity?

How are the intervals for parameter estimation defined? (pretty sure they are overlapping but that is about as far as I have got to with it).

if I'm using a dual-pair thevenin circuit model as defined by the powerpoint lecture notes, but the data does not include values for both, how should I work this out

Why is there only a detailed dataset for 25 degrees?

What method should be used for interpolation?
How best to handle any extrapolation (i.e. if the temp will exceed 40 deg)

For the sensitivity analysis, how should interpolation error be considered?

Where does the thermal/temperature data come from?
    How do I validate the thermal data?

# Code functionality
Basemodel.unpack(y) assumes the vector y is either made up by a flat list of scalars or will also work elementwise if arrays are passed in.
I used pack and unpack so that I didn't have to hardcode the amount of states in each model when I couple them up. it also makes pulling values from the state less prone to errors as I don't have to remember the indicie for a given state.

For parameter interpolation, all methods will default to linear unless specified otherwise, while adding in functionality to easily change it via the yaml files. This will be revisited once deeper into the project.