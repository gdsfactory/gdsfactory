from jumanji.environments.routing.connector.generator import Generator
from jumanji.environments.routing.connector.env import Connector
from jumanji.environments.routing.connector.utils import get_position, get_target
from jumanji.environments.routing.connector.types import State, Agent
from jumanji.wrappers import AutoResetWrapper
from jumanji.registration import register
from jumanji import make
import chex

import jax
import jax.numpy as np

import gdsfactory as gf

c = gf.Component()
obst1 = c << gf.components.straight(100)
obst2 = c << gf.components.straight(100)

obst1 = obst1.move(origin=(0, 0), destination=(0, 50))
obst2 = obst2.move(origin=(0, 0), destination=(50, 0))
# obst3 = c << gf.pcells.waveguide(10, 10, 0)
# obst3.transform(gf.kdb.Trans(0, False, 60 / c.klib.dbu, 30 / c.klib.dbu))

c.add_port(name="o1", center=(0, 0), width=.5, orientation=0, port_type="optical", layer="WG")
c.add_port(name="o2", center=(150, 0), width=0.5, orientation=180, port_type="optical", layer="WG")
c

class Grid(Generator):
    def __init__(self, c: gf.Component, ports1: list[gf.Port], ports2: list[gf.Port]):
        self.ports1 = ports1
        self.ports2 = ports2
        c_ = gf.Component()
        inst = c_ << c
        inst = inst.move(origin=(0, 0), destination=-c.center + (75, -25))
        c_.add_ports(inst.ports)
        self.c = c_
        grid_size = 150
        grid_size = int(grid_size)
        num_agents = len(ports1)

        super().__init__(grid_size, num_agents)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout.

        Returns:
            A `Connector` state.
        """
        key, pos_key = jax.random.split(key)
        starts_flat = np.array([], dtype=np.int32)
        for port in self.ports1:
            starts_flat = np.append(starts_flat, int(port.center[0] * port.center[1] - self.c.center[1]))
        targets_flat = np.array([], dtype=np.int32)
        for port in self.ports2:
            print(port.center, self.grid_size, self.c.ports["o2"])
            targets_flat = np.append(targets_flat, int(port.center[0] * port.center[1] + self.c.center[1]))

        # Create 2D points from the flat arrays.
        starts = np.divmod(starts_flat, self.grid_size)[::-1]
        targets = np.divmod(targets_flat, self.grid_size)
        # starts = (starts[0].astype(np.int32), starts[1].astype(np.int32))
        # targets = (targets[0].astype(np.int32), targets[1].astype(np.int32))
        print(starts, targets)
        # Get the agent values for starts and positions.
        agent_position_values = jax.vmap(get_position)(np.arange(self.num_agents))
        agent_target_values = jax.vmap(get_target)(np.arange(self.num_agents))

        # Create empty grid.
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Place the agent values at starts and targets.
        grid = grid.at[starts].set(agent_position_values)
        grid = grid.at[targets].set(agent_target_values)

        def get_obst_values(obst_id: np.int32) -> np.int32:
            return 1 + 3 * self.num_agents + 1

        agent_paths_values = jax.vmap(get_obst_values)(np.arange(self.num_agents))
        print(agent_paths_values)
        for shape in self.c.references:
            shape: gf.ComponentReference
            for point in shape.get_polygons():
              for polygon in point:
                polygon
                point_ = np.array([], dtype=np.int32)
                point_ = np.append(point_, np.array((int(polygon[0]), int(polygon[1]))))
                print(point_)
                print(grid.at[point_[0], point_[1]].get())
                grid = grid.at[point_[0], point_[1]].set(agent_paths_values[0])

        # Create the agent pytree that corresponds to the grid.
        print(starts, targets)
        agents = jax.vmap(Agent)(
            id=np.arange(self.num_agents),
            start=np.stack(starts, axis=1),
            target=np.stack(targets, axis=1),
            position=np.stack(starts, axis=1),
        )

        step_count = np.array(0, np.int32)

        return State(key=key, grid=grid, step_count=step_count, agents=agents)
    
class AutoRouter(Connector):
    def __init__(self, c, ports1, ports2):
        super().__init__(Grid(c, ports1, ports2))

# register("AutoRouter-v0", ".:AutoRouter", c=c, ports1=[c.ports["o1"]], ports2=[c.ports["o2"]])

env = Connector(generator=Grid(c, [c.ports["o1"]], [c.ports["o2"]]))
# env = Connector()
# env = AutoResetWrapper(env)     # Automatically reset the environment when an episode terminates
num_actions = env.action_spec().num_values

random_key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(random_key)

def step_fn(state, key, states):
  action = jax.random.randint(key=key, minval=0, maxval=num_actions, shape=(1,))
  new_state, timestep = env.step(state, action)
  states.append(new_state)
  return new_state, timestep, states
import jax

import jumanji
from jumanji.wrappers import AutoResetWrapper

env = AutoResetWrapper(env)     # Automatically reset the environment when an episode terminates

batch_size = 7
rollout_length = 5000
num_actions = env.action_spec().num_values

random_key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(random_key)

def step_fn(state, key):
  action = jax.random.randint(key=key, minval=0, maxval=num_actions, shape=(1,))
  new_state, timestep = env.step(state, action)
  return new_state, timestep

def run_n_steps(state, key, n):
  random_keys = jax.random.split(key, n)
  state, rollout = jax.lax.scan(step_fn, state, random_keys)
  return state, rollout

# Instantiate a batch of environment states
keys = jax.random.split(key1, batch_size)
state, timestep = jax.vmap(env.reset)(keys)

# Collect a batch of rollouts
keys = jax.random.split(key2, batch_size)
state, rollout = jax.vmap(run_n_steps, in_axes=(0, 0, None))(state, keys, rollout_length)

# Shape and type of given rollout:
# TimeStep(step_type=(7, 5), reward=(7, 5), discount=(7, 5), observation=(7, 5, 6, 6, 5), extras=None)

# Shape and type of given rollout:
# TimeStep(step_type=(7, 5), reward=(7, 5), discount=(7, 5), observation=(7, 5, 6, 6, 5), extras=None)
