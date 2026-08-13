import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import AffineTransform

var_dist = dist.TransformedDistribution(
    dist.Beta(2., 2.),
    [AffineTransform(loc=0., scale=1e-4)]
)

for i in range(10):
    var = pyro.sample("var", var_dist)
    print(var.item())