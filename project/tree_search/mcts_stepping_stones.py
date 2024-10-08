import os
import time
import numpy as np
from typing import List, Tuple
from itertools import product
import pickle

from environment.simulator import SteppingStonesSimulator
from tree_search.kinematics import QuadrupedKinematicFeasibility
from tree_search.abstract import MCTS, timing

# State is the current 4 contact locations, referenced by their indices
# State is now an np.int8 array with 4 elements
State = np.ndarray

class MCTSSteppingStonesKin(MCTS):
    PERF_TIME_FIRST = "time_first"
    PERF_NMPC_FIRST = "n_nmpc_first"
    PERF_IT_FIRST = "it_first"
    CONTACT_PLAN = "contact_plan"
    SIMULATION_TIME = "nmpc_time"
    GOAL = "goal"
    MCTS_PERF_FILE_NAME = "mcts_perf.pkl"
    
    def __init__(self,
                 stepping_stones_sim: SteppingStonesSimulator,
                 simulation_steps: int = 1,
                 C: float = np.sqrt(2),
                 W: float = 10.,
                 alpha_exploration: float = 0.0,
                 max_step_size: float | Tuple[float, float] = 0.24,
                 **kwargs,
                 ) -> None:
        
        self.sim = stepping_stones_sim
        self.alpha_exploration = alpha_exploration
        if isinstance(max_step_size, float):
            self.max_step_size_x, self.max_step_size_y = max_step_size, max_step_size * 0.65
        if isinstance(max_step_size, tuple):
            self.max_step_size_x, self.max_step_size_y = max_step_size, max_step_size
        self.C = C
        self.W = W
        
        optional_args = {
            "max_depth_selection" : 15,
            "max_solution_search" : 1,
            "max_trials_nmpc" : 50,
        }
        optional_args.update(kwargs)
        
        super().__init__(simulation_steps, C, **optional_args)
        
        # Function to shape reward
        sigmoid = lambda x : 1. / (1 + np.exp(-x))
        T = 5
        self.f = lambda x : sigmoid(T * (x - 1))
        
        # Algorithm performances. List for all solutions found
        self.performance = {
            MCTSSteppingStonesKin.PERF_TIME_FIRST : [],
            MCTSSteppingStonesKin.PERF_NMPC_FIRST : [],
            MCTSSteppingStonesKin.PERF_IT_FIRST : [],
            MCTSSteppingStonesKin.CONTACT_PLAN : [],
            MCTSSteppingStonesKin.GOAL : [],
            MCTSSteppingStonesKin.SIMULATION_TIME : [],
        }
        self.time_search_started = 0.
        self.nmpc_sim_count = 0
        
        # Maximum distance between contact locations
        self.d_max = self._compute_max_dist(self.sim.stepping_stones.positions)

    def _compute_max_dist(self, contact_pos_w) -> float:
        diffs = contact_pos_w[:, np.newaxis, :] - contact_pos_w[np.newaxis, :, :]
        d_squared = np.sum(diffs**2, axis=-1)
        d_max = np.sqrt(np.max(d_squared))
        return d_max

    @staticmethod
    def avg_dist_to_goal(contact_pos_w: np.ndarray,
                         current_states: List[State],
                         goal_state: State) -> float:
        """
        Computes average distance to goal.
        """
        d_to_goal = contact_pos_w[current_states] - contact_pos_w[np.newaxis, goal_state]
        avg_dist_to_goal = np.mean(np.linalg.norm(d_to_goal, axis=-1), axis=-1)
        return avg_dist_to_goal
    
    @timing("heuristic")
    def heuristic(self,
                  states: List[State],
                  goal_state: State) -> State:
        """
        Heuristic function to guide the search computed in a batched way.
        
        Args:
            states (List[State]): Compute the value of the heuristic on those states.
            goal_state (State): Goal state.

        Returns:
            State: State chosen by the heuristic.

        """
        heuristic_values = self.avg_dist_to_goal(
            self.sim.stepping_stones.positions,
            states,
            goal_state)

        # Exploration
        if np.random.rand() < self.alpha_exploration:
            probs = heuristic_values / sum(heuristic_values)
            id = np.random.choice(np.arange(len(states)), p=probs)

        # Exploitation
        else:
            id = np.argmin(heuristic_values)
        
        state = states[id]
        return state
    
    def sort_heuristic(self, states: List[State], goal_state : State, take_first : int):
        heuristic_values = self.avg_dist_to_goal(
            self.sim.stepping_stones.positions,
            states,
            goal_state)

        id = np.argsort(heuristic_values)[:take_first]
        state_sorted = states[id]
        return state_sorted
    
    def get_children(self, state: State) -> List[State]:
        """
        Get kinematically reachable states from the current state.

        Args:
            state (State): current state.

        Returns:
            List[State]: Reachable states as a list.
        """
        feet_pos_w = self.sim.stepping_stones.positions[state]

        # Shape [Nr, 4]
        max_x, max_y = self.sim.stepping_stones.spacing
        scale = 1.1 + self.sim.stepping_stones.randomize_pos_ratio / 2.
        possible_contact_id = [
            QuadrupedKinematicFeasibility.reachable_locations(
                foot_pos,
                self.sim.stepping_stones.positions,
                max_center_dist=(max_x * scale, max_y * scale),
            ) for foot_pos in feet_pos_w]

        # Combination of feet location [NComb, 4]
        all_states_id = np.array(list(product(*possible_contact_id)), dtype=np.int8)
        all_states_w = self.sim.stepping_stones.positions[all_states_id]

        # Prune feet contacts set that have a too high average displacement
        center_feet = np.mean(feet_pos_w, axis=0)
        center_possible_states = np.mean(all_states_w, axis=1)
        reachable = QuadrupedKinematicFeasibility.reachable_locations(
            center_feet,
            center_possible_states,
            max_center_dist=(self.max_step_size_x, self.max_step_size_y)
        )
        
        # Prune state that have crossing legs
        valid_states_id = all_states_id[reachable]
        valid_states_w = all_states_w[reachable]
        not_crossing = QuadrupedKinematicFeasibility.check_cross_legs(valid_states_w)
        
        legal_next_states = valid_states_id[not_crossing]
        # Take only the nodes with best heuristic (in direction of the goal) to speed up the search.
        legal_next_states_sorted = self.sort_heuristic(legal_next_states, self.state_goal, len(legal_next_states) // 3)
        
        return legal_next_states_sorted

    def reward(self,
               contact_plan: List[np.ndarray],
               goal_state: State,
               ) -> float:
        # If goal not reach: distance to goal
        if not np.array_equal(contact_plan[-1], goal_state):
            avg_d_goal = MCTSSteppingStonesKin.avg_dist_to_goal(
                self.sim.stepping_stones.positions,
                contact_plan[-1],
                goal_state,
            )[0]
            return self.f(1 - avg_d_goal / self.d_max)
        
        # If goal reach: dynamic feasibility
        sim_time = time.time()
        goal_reached = self.W * self.sim.run_contact_plan(contact_plan, verbose=False, use_viewer=False, real_time=False)
        sim_time = time.time() - sim_time
        self.performance[MCTSSteppingStonesKin.SIMULATION_TIME].append(sim_time)
        self.nmpc_sim_count += 1
        
        # Stop simulation in case of too many trials (unfeasible)
        if (self.max_trials_nmpc > 0 and
            self.nmpc_sim_count >= self.max_trials_nmpc):
            self.max_solution_search = -1
            
        return goal_reached
    
    @staticmethod
    def repeat_first_contact_plan(contact_plan, n_repeat: int = 0):
        """
        Repeat first jump of a contact plan.
        """
        contact_plan_ext = [contact_plan[0]] * n_repeat + contact_plan
        return contact_plan_ext

    @timing("simulation")
    def simulation(self, state: State, goal_state: State) -> float:
        
        simulation_path = []
        for _ in range(self.simulation_steps):
            
            # Choose successively one child until goal is reached
            if self.tree.has_children(state) and not self.is_terminal(state, goal_state):
                
                children = self.tree.get_children(state)
                state = self.heuristic(children, goal_state)

                simulation_path.append(state)
            else:
                break
            
        contact_plan = self.tree.current_search_path + simulation_path
        contact_plan = self.repeat_first_contact_plan(contact_plan, 2)
        
        reward = self.reward(contact_plan, goal_state)
        solution_found = reward >= 1
        
        if solution_found:
            self._record_search_performance(contact_plan)
            self.all_solutions.append(contact_plan)

                        
        return reward, solution_found
    
    def search(self, num_iterations: int = 10000):
        self.state_start = np.array(self.sim.start_indices, dtype=np.int8)
        self.state_goal = np.array(self.sim.goal_indices, dtype=np.int8)
        self.time_search_started = time.time()
        self.performance[MCTSSteppingStonesKin.GOAL] = self.state_goal
        
        super().search(self.state_start, self.state_goal, num_iterations)
    
    def _record_search_performance(self, contact_plan) -> None:
        """
        Save search performance in a dict {perf name : [perf for each solutions found]}
        - Time to jth solution found
        - Iteration jth first solution found
        - MPC simulation to jth solution found
        - contact plan found of the jth solution found
        """
        # Record timings when 1st solution is found
        current_time = time.time()
        time_to_find_solution = (
            current_time - 
            sum(self.performance[MCTSSteppingStonesKin.PERF_TIME_FIRST]) -
            self.time_search_started
            )
        
        if len(self.all_solutions) > 0:
            it_to_find_solution = self.it - self.performance[MCTSSteppingStonesKin.PERF_IT_FIRST][-1]
            nmpc_sim_count_to_find_solution = self.nmpc_sim_count - self.performance[MCTSSteppingStonesKin.PERF_NMPC_FIRST][-1]

        else:
            it_to_find_solution = self.it
            nmpc_sim_count_to_find_solution = self.nmpc_sim_count
        
        self.performance[MCTSSteppingStonesKin.PERF_TIME_FIRST].append(time_to_find_solution)
        self.performance[MCTSSteppingStonesKin.PERF_IT_FIRST].append(it_to_find_solution)
        self.performance[MCTSSteppingStonesKin.PERF_NMPC_FIRST].append(nmpc_sim_count_to_find_solution)
        self.performance[MCTSSteppingStonesKin.CONTACT_PLAN].append(contact_plan)
    
    def save_search_performances(self, saving_dir : str) -> None:
        if not os.path.exists(saving_dir): os.makedirs(saving_dir)
        file_path = os.path.join(saving_dir, MCTSSteppingStonesKin.MCTS_PERF_FILE_NAME)
        
        # Saving a dictionary to a Pickle file
        with open(file_path, 'wb') as f:
            pickle.dump(self.performance, f)