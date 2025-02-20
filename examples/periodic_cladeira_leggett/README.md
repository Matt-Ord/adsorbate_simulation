## Periodic Caldeira Leggett Environment Examples

In this folder there are a series of example demonstrating simulations using the Periodic Caldeira-leggett model

In contrast to the `caldeira_leggett` method, this uses a periodic interation
to approximate the system-environment interaction. These are plotted in `environment_operators.py`,
we have two such operators; a `sin x` and a `cos x` operator.

In `stochastic_schrodinger.py` we have a simple example of simulating a free particle.
The thermal behavior is tested in `thermal_occupation.py` and `thermal_investigation.py`,
as a useful test of convergence of the model. The displacements can be directly compared
to the langevin equation (see `displacements.py`), and for a free particle these
should match the analytical formula. The states are approximately gaussian states,
and should eb evenly distributed in space. The exact distribution is analzed in
`gaussian_states.py`.
