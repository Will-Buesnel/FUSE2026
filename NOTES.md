## Notes on Lab data:*
MLP001_wltp_25degC.xlsx: raw data from our battery cycler, where the current has been imposed and the voltage has been measured
wltp.csv: the current that was imposed. This is a drive-cycle file, giving current as a function of time. This plus the voltage data are useful for validating battery models.
MLP001_params.csv: resistance and timescale parameters for a cell, for various states of charge and temperatures
MLP001_ocv.csv: open-circuit voltage data for the same cell

*Lab data is currently gitignored.


Reproducing fig 11. :  
Fig. 11 compares model and experimental data on a drive-cycle with highly variable currents, representative of the power demands on an electric vehicle. Notably, the currents throughout the drive cycle are different to those used for parameterisation; this will increase the model error, but offers a useful test of how well the model performs on unseen data. A capacity of 2.13 Ah is used in the model, which reflects capacity losses in the cell after the parameterisation and before the validation experiments. The model generally performs well, and model errors compare favourably to literature results