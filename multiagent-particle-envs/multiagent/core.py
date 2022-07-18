import numpy as np


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None

        # MSCLAR: physical position in cell grids, only useful for RealGridWorld
        # TODO: improve object hierarchy!
        self.p_pos_grid = None

        # physical velocity
        self.p_vel = None


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None


# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()


# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

        self.num_cells_hearing = None
        self.num_cells_visibility = None

        # bool array of same length for all agents with the first hand information they have
        self.first_hand_information = None


class Wall(Entity):
    def __init__(self, cell_size):
        super(Wall, self).__init__()
        # properties:
        self.size = cell_size
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 2500000
        # mass
        self.initial_mass = 10000000


class SpecialCell(Entity):
    def __init__(self, cell_size):
        """
        Ghost space of a specific color, used to refresh the
        information when someone of that colors steps in it.
        """
        super(SpecialCell, self).__init__()
        # properties:
        self.size = cell_size
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = False


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.walls = []

        self.collaborative = None  # default value (should be assigned True / False in scenarios)

        # i-th element contains all the information agent i has heard so far
        self.knowledge_storage = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks + self.walls

    @property
    def entities_without_walls(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def process_p_force(self):
        # gather forces applied to entities
        p_force = [None] * len(self.entities_without_walls)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        self.process_p_force()

        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities_without_walls):
            for b in range(a + 1, len(self.entities_without_walls)):
                entity_b = self.entities[b]
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]

        return p_force

    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities_without_walls):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(
                                                                          entity.state.p_vel[1])) * entity.max_speed

            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]


class RealGridWorld(World):
    def __init__(self, cells_width):
        super(RealGridWorld, self).__init__()

        self.special_cells = []

        self.listening_reward = True
        self.speaking_reward = True
        self.cells_width = cells_width  # number of tiles in width and height (assuming squares)
        self.grid = [[[] for _ in range(cells_width)] for _ in range(cells_width)]
        self.restricted_speech = False
        self.first_hand_info_only = True  # to be compatible with the initial setting I used

        # knowledge_storage[i][j] is True iff agent i knows information j
        self.knowledge_storage = None

        # tmp_knowledge_storage to update all knowledge_storage at the same time
        # without this, I cannot reward speakers and listeners symmetrically
        self.tmp_knowledge_storage = None

    @property
    def entities(self):
        return self.agents + self.landmarks + self.walls + self.special_cells

    @property
    def tile_size(self):
        return 2 / self.cells_width

    def r2_position_to_gridcell(self, agent_state_p_pos):
        """
        Convert R2 position in the plane to the grid pair [0, cells_width) x [0, cells_width).
        Currently used only for efficiency purposes in observation().
        R2 cells are (minimum_pos + tile_size * np.random.randint(1, cells_width - 1, size=world.dim_p))

        Note: temporary fix until I change everything to grid coordinates.
        """
        minimum_pos = -1 + self.tile_size / 2
        return np.rint((agent_state_p_pos - minimum_pos) / self.tile_size).astype(int)

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        self.knowledge_storage = self.tmp_knowledge_storage.copy()
        self.process_p_force()

        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):

        # entity vs. walls
        for a, entity_a in enumerate(self.entities_without_walls):
            next_pos = entity_a.state.p_pos + (p_force[a] / entity_a.mass) * self.dt * self.dt
            next_pos_grid = self.r2_position_to_gridcell(next_pos)

            if any(name == 'wall' for name, idx in self.grid[next_pos_grid[0]][next_pos_grid[1]]):
                p_force[a] = np.array([0, 0])

        # simple (but inefficient) collision response between agents
        # the extra loop is for cases where there are multiple agents moving at the same time,
        # and the old position is now invalid because someone else is trying to move there
        cancelled_movement = True
        while cancelled_movement:
            cancelled_movement = False
            for a, entity_a in enumerate(self.agents):
                for b in range(a + 1, len(self.agents)):
                    entity_b = self.agents[b]

                    next_pos_entity_a = entity_a.state.p_pos + (p_force[a] / entity_a.mass) * self.dt * self.dt
                    next_pos_entity_b = entity_b.state.p_pos + (p_force[b] / entity_b.mass) * self.dt * self.dt

                    next_pos_entity_a = self.r2_position_to_gridcell(next_pos_entity_a)
                    next_pos_entity_b = self.r2_position_to_gridcell(next_pos_entity_b)

                    # if smaller than minimum allowed distance, abort movement for one of the agents
                    # but making sure that the agent had movement this turn!
                    if not (next_pos_entity_a - next_pos_entity_b).any():
                        agent_ids = [a, b]
                        np.random.shuffle(agent_ids)
                        for agent_id in agent_ids:
                            if p_force[agent_id].any():
                                p_force[agent_id] = np.array([0, 0])
                                cancelled_movement = True
                                break

        return p_force

    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities_without_walls):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(
                                                                          entity.state.p_vel[1])) * entity.max_speed

            entity.state.p_pos += entity.state.p_vel * self.dt
            entity.state.p_pos_grid = self.r2_position_to_gridcell(entity.state.p_pos)
