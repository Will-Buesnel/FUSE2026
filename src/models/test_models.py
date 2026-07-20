"Will Buesnel, Jul 26."

from parameters import load_parameter_interpolants
from electrical import ElectricalModel

def test_simulation():
    # load the parameter interpolants from the CSV file
    param_file_path = "data/processed/MLP001_params.csv"
    param_interpolants = load_parameter_interpolants(param_file_path)
    print("Loaded parameter interpolants:")
    for param_name, interpolant in param_interpolants.items():
        print(f"{param_name}: {interpolant}")
    
    # create an instance of the ElectricalModel with the loaded parameters
    model_params = {
        "R_cell": param_interpolants["r0"],
        "SoC": 1,  # initial state of charge
        "Voc": 3.7,  # open-circuit voltage
        "ambient_temp": 25.0,  # ambient temperature in Celsius
        "T": 25.0,  # current temperature of the cell
        "current_func": lambda t: 1.0,  # constant current of 1A
        "max_capacity": 2.0,  # nominal capacity in Ah
    }
    model = ElectricalModel(model_params)

    # set the interpolants as funcs for the model
    model.set_c1_func(param_interpolants["c1"])
    model.set_c2_func(param_interpolants["c2"])
    model.set_r0_func(param_interpolants["r0"])
    model.set_r1_func(param_interpolants["r1"])
    model.set_r2_func(param_interpolants["r2"])

    # simulate the model for a given time span
    t_max = 3600  # simulate for 1 hour
    max_step = 1.0  # maximum step size for the solver
    atol = 1e-6  # absolute tolerance for the solver
    rtol = 1e-3  # relative tolerance for the solver
    model.simulate(
        current_func=model_params["current_func"],
        max_capacity=model_params["max_capacity"],
        initial_soc=model_params["SoC"],
        ambient_temp=model_params["ambient_temp"],
        t_max=t_max,
        max_step=max_step,
        atol=atol,
        rtol=rtol,
        pbar=True  # enable progress bar
    )


def test_rhs_with_interpolants(): # currently the interpolants for v and 
    # load the parameter interpolants from the CSV file
    param_file_path = "data/processed/MLP001_params.csv"
    param_interpolants = load_parameter_interpolants(param_file_path)
    
    # create an instance of the ElectricalModel with the loaded parameters
    model_params = {
        "R_cell": param_interpolants["r0"],
        "SoC": 1,  # initial state of charge
        "Voc": 3.7,  # open-circuit voltage
        "ambient_temp": 25.0,  # ambient temperature in Celsius
        "T": 25.0,  # current temperature of the cell
        "current_func": lambda t: 1.0,  # constant current of 1A
        "max_capacity": 2.0,  # nominal capacity in Ah
    }
    model = ElectricalModel(model_params)
    model.max_capacity=model_params["max_capacity"]
    # set the interpolants as funcs for the model
    model.set_c1_func(param_interpolants["c1"])
    model.set_c2_func(param_interpolants["c2"])
    model.set_r0_func(param_interpolants["r0"])
    model.set_r1_func(param_interpolants["r1"])
    model.set_r2_func(param_interpolants["r2"])

    # test the _rhs function at t=0 with initial state
    t = 0.0
    y = [model_params["SoC"], 0.0, 0.0]  # initial state: [soc, v_rc1, v_rc2]
    dy_dt = model._rhs(t, y, model_params["current_func"], model_params["T"], verbose=True)
    print("dy/dt at t=0:", dy_dt)

if __name__ == "__main__":
    test_rhs_with_interpolants()