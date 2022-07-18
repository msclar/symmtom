import numpy as np
from multiagent.core import RealGridWorld, Agent, Wall, SpecialCell
from multiagent.scenario import BaseScenario

COLORS = [
    [0.75, 0.25, 0.25],  # red
    [0.25, 0.75, 0.25],  # green
    [0.25, 0.25, 0.75],  # blue
    [0.75, 0.75, 0.15],  # yellow / orange
    [0.75, 0.15, 0.75],  # magenta
    [0.15, 0.75, 0.75],  # cyan
    [0.35, 0.05, 0.35],  # purple
    [0.35, 0.05, 0.05],  # maroon
    [0.05, 0.35, 0.35],  # teal
    [0.35, 0.35, 0.05],  # olive
]

GRIDWORLD_DIRECTIONS = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]


def get_normalized_agent_directions(agent, world):
    normalize = lambda vel: vel * world.cells_width / 2 * world.dt
    result = [normalize(agent.state.p_vel)]
    for other in world.agents:
        if other is agent:
            continue
        result.append(normalize(other.state.p_vel))
    return result


def initialize_world_agents(world, num_agents, num_cells_visibility, num_cells_hearing):
    world.agents = [Agent() for _ in range(num_agents)]
    for i, agent in enumerate(world.agents):
        agent.noise = 0
        agent.agent_id = i  # added only to have better performance when looping through world.agents
        agent.name = 'agent %d' % i
        agent.accel = (1 / world.dt) * (1 / world.dt) * 2 / world.cells_width  # amplifies movement
        agent.num_cells_visibility = num_cells_visibility if num_cells_visibility else world.cells_width
        agent.num_cells_hearing = num_cells_hearing if num_cells_hearing else world.cells_width
        agent.collide = False  # collision works differently because it's handled by RealGridWorld()


def initialize_outer_walls(world):
    world.walls = [Wall(1 / world.cells_width) for _ in range(world.cells_width * 4 - 4)]
    for wall in world.walls:
        wall.name = 'wall'
        wall.movable = False
        wall.color = np.array([0, 0, 0])


def other_agents_communication_vector(world, agent):
    # communication of all other agents: you can only hear what is close to you
    tile_size = 2 / world.cells_width
    comm = []
    for other in world.agents:
        if other is agent:
            continue
        distance = np.max(np.abs(other.state.p_pos - agent.state.p_pos))
        if distance < (agent.num_cells_hearing - 1) * tile_size + 1e-6:
            comm.append(other.state.c)
        else:
            comm.append(np.zeros(world.dim_c))
    return comm


def get_agents_positions_and_colors(world, agent):
    # get positions of all entities in this agent's reference frame
    tile_size = 2 / world.cells_width  # because the rendering is [-1, 1]

    entity_pos = []
    entity_colors = []
    for other in world.agents:
        if other is agent:
            continue
        entity_pos.append((other.state.p_pos - agent.state.p_pos) / tile_size / (world.cells_width - 3))
        entity_colors.append(other.color)
    return entity_pos, entity_colors


def new_information_reward(agent_id, other_agent_c, world):
    first_hand_info = world.agents[agent_id].first_hand_information

    reward = 0
    logging = {'first_hand_known_info_heard': 0, 'second_hand_known_info_heard': 0, 'known_info_heard': 0}
    for j, v in enumerate(other_agent_c):
        if 1 - 1e-6 < v < 1 + 1e-6:
            if world.knowledge_storage[agent_id][j]:
                logging['known_info_heard'] += 1
                logging['first_hand_known_info_heard'] += int(first_hand_info[j])
                logging['second_hand_known_info_heard'] += 1 - int(first_hand_info[j])
            else:
                reward += 1
                world.tmp_knowledge_storage[agent_id][j] = True
    return reward, logging


def reward_no_hearing_interference(agent, world):
    reward = 0
    logging = {'first_hand_known_info_heard': 0, 'second_hand_known_info_heard': 0, 'known_info_heard': 0,
               'listening_reward': 0, 'speaking_reward': 0}
    for i, other in enumerate(world.agents):
        if agent.agent_id == i:
            continue
        distance = np.max(np.abs(other.state.p_pos - agent.state.p_pos))
        if distance < (agent.num_cells_hearing - 1) * world.tile_size + 1e-6:
            if world.listening_reward:
                cur_reward, cur_log_info_heard = new_information_reward(agent.agent_id, other.state.c, world)
                reward += cur_reward
                logging['listening_reward'] += cur_reward
                for k, v in cur_log_info_heard.items():
                    logging[k] += v

            if world.speaking_reward:
                cur_reward = new_information_reward(i, agent.state.c, world)[0]
                reward += cur_reward
                logging['speaking_reward'] += cur_reward

    return reward, logging


def get_random_cellgrid_positions(world, num_agents, ratio=10):
    result = set([])
    repetitions = 0
    while len(result) < num_agents and repetitions < num_agents * ratio:
        # FIXME: only works for world.dim_p = 2 (2D environment)
        candidate = [int(t) for t in np.random.randint(1, world.cells_width - 1, size=2)]
        is_there_wall = any(name == 'wall' for name, _ in world.grid[candidate[0]][candidate[1]])
        if not is_there_wall:
            result.add(tuple(candidate))
        repetitions += 1

    result = list(result)
    np.random.shuffle(result)
    return result


def print_agents(world):
    grid = [['-' for _ in range(world.cells_width)] for _ in range(world.cells_width)]
    for agent in world.agents:
        char = chr((ord('A') if agent.state.c.max() > 1 - 1e-6 else ord('a')) + agent.agent_id)
        grid[agent.state.p_pos_grid[0]][agent.state.p_pos_grid[1]] = char

    for row in grid:
        print("".join(row))
    print()


def print_grid(world):
    grid = [['-' for _ in range(world.cells_width)] for _ in range(world.cells_width)]
    for x in range(world.cells_width):
        for y in range(world.cells_width):
            if any(a == 'wall' for a, _ in world.grid[x][y]):
                grid[x][y] = 'X'
            if any(a == 'info_base' for a, _ in world.grid[x][y]):
                grid[x][y] = chr(ord('A') + next(b for a, b in world.grid[x][y] if a == 'info_base'))

    for row in grid:
        print("".join(row))
    print()


def add_outer_walls_positions(world):
    tile_size = world.tile_size  # because the rendering is [-1, 1]
    minimum_pos = -1 + tile_size / 2

    cur = 0
    for i in range(world.cells_width - 1):
        for x, y in [(0, i), (i, world.cells_width - 1), (world.cells_width - 1, i + 1), (i + 1, 0)]:
            world.walls[cur].state.p_pos_grid = np.array([x, y], dtype=int)
            world.walls[cur].state.p_pos = np.array([minimum_pos + tile_size * x, minimum_pos + tile_size * y])
            world.grid[x][y].append(('wall', cur))
            cur += 1

    return cur


def initialize_big_middle_block(world, size_ratio=0.5):
    """
    Block will be in the center of the field, occupying size_ratio of the available field.
    If world.cells_width = 10, that leaves an 8 x 8 block of available pieces, and the big middle block will
    be of (world.cells_width - 2) * size_ratio length.
    """
    if not world.big_middle_block:
        return

    length = int(np.round((world.cells_width - 2) * size_ratio))
    for _ in range(length * length):
        wall = Wall(1 / world.cells_width)
        wall.name = 'wall'
        wall.movable = False
        wall.color = np.array([0, 0, 0])
        world.walls.append(wall)


def add_big_middle_block(world, cur_idx, size_ratio=0.5):
    if not world.big_middle_block:
        return

    tile_size = 2 / world.cells_width  # because the rendering is [-1, 1]
    minimum_pos = -1 + tile_size / 2
    length = int(np.round((world.cells_width - 2) * size_ratio))

    lowest_x = world.cells_width // 2 - length // 2
    for x in range(lowest_x, lowest_x + length):
        for y in range(lowest_x, lowest_x + length):
            world.walls[cur_idx].state.p_pos_grid = np.array([x, y], dtype=int)
            world.walls[cur_idx].state.p_pos = np.array([minimum_pos + tile_size * x, minimum_pos + tile_size * y])
            world.grid[x][y].append(('wall', cur_idx))
            cur_idx += 1


def reset_agents(world, agent_positions, first_hand_info, colors):
    minimum_pos = -1 + world.tile_size / 2

    world.knowledge_storage = np.zeros((len(world.agents), world.dim_c), dtype=bool)
    world.tmp_knowledge_storage = np.zeros((len(world.agents), world.dim_c), dtype=bool)
    world.accum_reward = [0 for _ in range(len(world.agents))]

    for i, agent in enumerate(world.agents):
        agent.first_hand_information = np.zeros(world.dim_c, dtype=bool)
        agent.first_hand_information[first_hand_info[i]] = True
        agent.color = colors[i]

        # 1 / self.world.cells_width would make the agent occupy whe whole grid cell, and we use 80% of the space
        agent.size = 0.8 / world.cells_width

        agent.state.p_pos_grid = np.array(agent_positions[i])
        agent.state.p_pos = minimum_pos + world.tile_size * agent.state.p_pos_grid
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)

        # you already know your own information
        # True only on first hand info, implemented like this to not clone arrays
        if world.restricted_speech:
            world.tmp_knowledge_storage[i][first_hand_info[i]] = True
            world.knowledge_storage[i][first_hand_info[i]] = True


def reset_information_bases(world, info_base_positions, colors):
    minimum_pos = -1 + world.tile_size / 2

    for i, cell in enumerate(world.special_cells):
        cell.state.p_pos_grid = np.array(info_base_positions[i])
        cell.state.p_pos = minimum_pos + world.tile_size * cell.state.p_pos_grid
        cell.color = colors[i]
        world.grid[info_base_positions[i][0]][info_base_positions[i][1]].append(
            ('info_base', i))  # FIXME: this name cannot be changed


def np_array_equal(a, b):
    return np.max(np.abs(a - b)) < 1e-6


class Scenario(BaseScenario):

    def __init__(self):
        self.proportional_reward = False
        self.constant_reward = None

    def make_world(self, dim_c=5, num_agents=2, cells_width=10,
                   num_cells_hearing=3, num_cells_visibility=None,
                   listening_reward=True, speaking_reward=True, restricted_speech=False,
                   proportional_reward=False, constant_reward=None, movement_penalty=0,
                   hearing_interference=False, big_middle_block=False, upperbound=False,
                   zeroth_info_always_given=False):
        """
        Definition of the scenario (similar to init, once per episode).

        **VARIABLE PARAMETERS IN CURRENT DEFINITION OF SYMMTOM**

        dim_c: Number of communicative pieces in total
        num_agents: Number of agents
        cells_width: 2 + width of the desired squared environment.
            It is +2 since it is also counting the outside border.
        upperbound: True iff running an oracle run (perfect information for all agents).


        **PARAMETERS ALWAYS FIXED IN CURRENT DEFINITION OF SYMMTOM (they could be modified to change difficulty.)**

        num_cells_hearing: Width of the square centered in the agent. =3 for all experiments, which means only
            immediate neighbors can listen.
        num_cells_visibility: Can be used to restrict agent vision. =None for all experiments (perfect vision).
        listening_reward: =True for all experiments.
        speaking_reward: =True for all experiments.
        restricted_speech: Only speak what you know. =True in all experiments.


        **ADDITIONAL MODIFICATIONS TO SYMMTOM, UNUSED IN CURRENT EXPERIMENTS**

        big_middle_block: Version of the environment where there is a big middle block of inaccessible cells. Unused
            in current experiments, but can serve as a more difficult version of the environment, since distances
            are longer.
        zeroth_info_always_given: True means own information is always given to each agent (aka zeroth
            theory of mind will not be required). =False for all experiments.
        hearing_interference: An agent listening two different agents at the same time will not be able to learn
            anything (similar to when two people speak to you at the same time :)). =False for all experiments.
        movement_penalty: Penalty for moving. =0 in all experiments.
        proportional_reward & constant_reward: explorations on different rewards, unused in current experiments.

        """

        self.proportional_reward = proportional_reward
        self.constant_reward = constant_reward

        world = RealGridWorld(cells_width)
        world.upperbound = upperbound
        world.zeroth_info_always_given = zeroth_info_always_given
        world.num_agents_per_type = 1  # MSCLAR: only for this scenario
        world.restricted_speech = restricted_speech
        world.first_hand_info_only = False
        world.movement_penalty = movement_penalty
        world.hearing_interference = hearing_interference
        world.big_middle_block = big_middle_block

        world.discrete_action = True
        world.collaborative = False
        world.damping = 1.0
        world.dim_c = dim_c
        world.listening_reward = listening_reward
        world.speaking_reward = speaking_reward
        world.cells_width = cells_width  # number of tiles in width and height (assuming squares)

        # MSCLAR: for debugging purposes, should be <= listening_rew * dim_c + speaking_rew * dim_c
        world.accum_reward = [0 for _ in range(len(world.agents))]

        # add agents
        initialize_world_agents(world, world.num_agents_per_type * num_agents, num_cells_visibility, num_cells_hearing)
        initialize_outer_walls(world)
        initialize_big_middle_block(world)

        world.special_cells = [SpecialCell(1 / world.cells_width) for _ in range(num_agents)]
        for i, cell in enumerate(world.special_cells):
            cell.name = 'info_base %d' % i

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, seed=None):
        if seed:
            np.random.seed(seed)
        world.grid = [[[] for _ in range(world.cells_width)] for _ in range(world.cells_width)]

        # make a list where all information bits appeared balanced (although not always the same amount of times)
        # then we shuffle that list and assign sequentially
        # dirichlet_colors = np.random.rand(len(world.agents), 3)
        dirichlet_colors = np.array(COLORS[:len(world.agents)])

        # extra elems are the ones that do not fit in mod num_agents, so we add in a balanced way to reach dim_c
        extra_elements = np.arange(len(world.agents))
        np.random.shuffle(extra_elements)
        extra_elements = extra_elements[:(world.dim_c % len(world.agents))]

        info_ids = np.concatenate([np.arange(len(world.agents)) for _ in range(world.dim_c // len(world.agents))])
        info_ids = np.concatenate([info_ids, extra_elements])
        np.random.shuffle(info_ids)

        dirichlet_info = [[] for _ in range(len(world.agents))]
        for i, agent_id in enumerate(info_ids):
            dirichlet_info[agent_id].append(i)

        cur_idx = add_outer_walls_positions(world)
        add_big_middle_block(world, cur_idx)

        # add special cells for information bases, one per color
        info_base_positions = get_random_cellgrid_positions(world, world.dim_c)
        reset_information_bases(world, info_base_positions, dirichlet_colors)

        cellgrid_positions = get_random_cellgrid_positions(world, len(world.agents), ratio=1000)
        agent_types = np.array([np.arange(world.dim_c) for _ in range(world.num_agents_per_type)]).reshape(-1)
        np.random.shuffle(agent_types)

        # there are dim_c pieces of information, unknown at first
        reset_agents(world, cellgrid_positions, dirichlet_info, dirichlet_colors)

    def preprocess_rewards(self, world):
        if not world.hearing_interference:
            return

        world.heard_information = np.zeros([len(world.agents), world.dim_c])
        for i, agent in enumerate(world.agents):
            for j, other in enumerate(world.agents):
                if i == j:
                    continue
                distance = np.max(np.abs(other.state.p_pos - agent.state.p_pos))
                if distance < (agent.num_cells_hearing - 1) * world.tile_size + 1e-6:
                    indexes = np.arange(len(other.state.c))[
                        np.logical_and(1 - 1e-6 < other.state.c, other.state.c < 1 + 1e-6)]
                    world.heard_information[i][indexes] += 1

        # number of speakers each agent hears, > 1 is too much information and nothing will be processed
        world.concurrent_speakers = world.heard_information.sum(axis=1)

    def reward(self, agent, world):
        """
        Reward based on hearing new information or giving out new information to other agents.
        """

        # for logging purposes, expectation is that this value should decrease as agents are optimized
        logging = {'first_hand_known_info_heard': 0, 'second_hand_known_info_heard': 0, 'known_info_heard': 0,
                   'listening_reward': 0, 'speaking_reward': 0, 'recharge_base_reward': 0,
                   'correct_recharge_base_use': 0, 'incorrect_recharge_base_use': 0}
        reward = 0

        # penalty for having any movement
        if np.max(np.abs(agent.state.p_vel)) > 1e-6:
            reward -= world.movement_penalty

        if world.hearing_interference:
            if world.listening_reward:
                info_heard_current_turn = world.heard_information[agent.agent_id] > 0
                if world.concurrent_speakers[agent.agent_id] == 1:
                    world.tmp_knowledge_storage[agent.agent_id][info_heard_current_turn] = True

                    # note that there can only be at most one non-zero value (eq to 1) because of interference
                    info_idx = world.knowledge_storage[agent.agent_id][info_heard_current_turn].argmax()

                    known_info_heard = int(world.knowledge_storage[agent.agent_id][info_idx])  # zero or one
                    new_info_heard = 1 - known_info_heard
                    is_first_hand_info = int(agent.first_hand_information[info_idx])
                    is_second_hand_info = 1 - is_first_hand_info

                    reward += new_info_heard
                    logging['listening_reward'] += new_info_heard
                    logging['known_info_heard'] += known_info_heard
                    logging['first_hand_known_info_heard'] += known_info_heard * is_first_hand_info
                    logging['second_hand_known_info_heard'] += known_info_heard * is_second_hand_info

            if world.speaking_reward:
                for i, other in enumerate(world.agents):
                    # TODO: check this if statement
                    if agent.agent_id == i or world.concurrent_speakers[i] > 1:
                        continue
                    distance = np.max(np.abs(other.state.p_pos - agent.state.p_pos))
                    if distance < (agent.num_cells_hearing - 1) * world.tile_size + 1e-6:
                        rew = new_information_reward(i, agent.state.c, world)[0]
                        reward += rew
                        logging['speaking_reward'] += rew
        else:
            cur_reward, cur_log = reward_no_hearing_interference(agent, world)
            reward += cur_reward
            for k, v in cur_log.items():
                logging[k] += v

        # big reward and forget all second-hand information when hitting your own base
        # this only happens if you have know all the information
        all_nothing_mode = self.constant_reward is None and not self.proportional_reward
        all_knowledge = world.knowledge_storage[agent.agent_id].all()
        if not all_nothing_mode or all_knowledge:
            for name, idx in world.grid[agent.state.p_pos_grid[0]][agent.state.p_pos_grid[1]]:
                if name == 'info_base' and np_array_equal(agent.color, world.special_cells[idx].color):
                    rew = self.constant_reward if self.constant_reward is not None else \
                        world.knowledge_storage[agent.agent_id].sum() * (len(world.agents) - 1)
                    reward += rew
                    logging['recharge_base_reward'] += rew
                    logging['correct_recharge_base_use'] += 1

                    # constant rewards are not supposed to make you forget anything, just reward for going there
                    if self.constant_reward is None:
                        world.tmp_knowledge_storage[agent.agent_id] = agent.first_hand_information.copy()

        # log incorrect recharge base use without modifying code above
        if all_nothing_mode and not all_knowledge:
            for name, idx in world.grid[agent.state.p_pos_grid[0]][agent.state.p_pos_grid[1]]:
                in_own_base = (name == 'info_base' and np_array_equal(agent.color, world.special_cells[idx].color))
                if in_own_base:
                    logging['incorrect_recharge_base_use'] += 1

        # Note: talking may be restricted to only the first_hand_information, see environment.py:_set_action()
        world.accum_reward[agent.agent_id] += reward
        return reward, logging

    def observation(self, agent, world):
        entity_pos, _ = get_agents_positions_and_colors(world, agent)

        # inform which of the immediate four directions are walls
        walls = []
        for dir in GRIDWORLD_DIRECTIONS:
            neighbor = world.r2_position_to_gridcell(agent.state.p_pos) + dir
            walls.append(any(name == 'wall' for name, _ in world.grid[neighbor[0]][neighbor[1]]))

        comm = other_agents_communication_vector(world, agent)

        # normalized position of all recharge bases for that agent (for now, only one)
        tile_size = 2 / world.cells_width  # because the rendering is [-1, 1]
        base_locations = []
        for cell in world.special_cells:
            if np_array_equal(agent.color, cell.color):
                base_locations.append((cell.state.p_pos - agent.state.p_pos) / tile_size / (world.cells_width - 3))

        # recharge base position for all other agents
        for other in world.agents:
            if other is agent:
                continue
            for cell in world.special_cells:
                if np_array_equal(other.color, cell.color):
                    base_locations.append((cell.state.p_pos - other.state.p_pos) / tile_size / (world.cells_width - 3))

        tmp = []
        if world.restricted_speech:
            if world.zeroth_info_always_given or world.upperbound:
                tmp += [world.knowledge_storage[agent.agent_id]]
            tmp += [agent.first_hand_information]
        tmp += get_normalized_agent_directions(agent, world) + entity_pos

        # first hand information are "explicit" colors (colors are a good injective function)
        if world.upperbound:
            tmp += [world.knowledge_storage[other.agent_id] for other in world.agents if other is not agent]
        tmp += [other.first_hand_information for other in world.agents if other is not agent]
        tmp += base_locations + comm + [walls]
        return np.array([np.concatenate(tmp)])

    def benchmark_data(self, agent, world):
        # tmp_knowledge_storage is more up-to-date than the non-tmp version (updated every turn)
        return world.tmp_knowledge_storage[agent.agent_id].sum()

    def done(self, agent, world):
        # agent has heard everything it can learn
        return False
