import tyro
import numpy as np

from utils.eval import PerformanceGatherer


class MCTSPerformancesEvaluation():
    def __init__(self, experiment_dir : str) -> None:
        self.exp_dir = experiment_dir
        self.perfs = PerformanceGatherer(experiment_dir)
        self.data_perfs = self.perfs.data

        self.compute_statistics_first_sol()
        self.compute_statistics_contact_plans()
        self.compute_success()
        self.get_longest_contact_plans(10)
    
    def _compute_statistics_list(self, l):
        mean = np.mean(l)
        std = np.std(l)
        median = np.median(l)
        min = np.min(l)
        max = np.max(l)
        
        print("Mean", mean)
        print("Std", std)
        print("Median", median)
        print("Min", min)
        print("Max", max)
        print()
        
    def _compute_statistics_first_sol(self, key:str):
        if key in self.data_perfs.keys():
            data = self.data_perfs[key]
            
            try:
                print("---", key, "first solution found")
                data_first_sol = [sol[0] for sol in data if len(sol) > 0]
                self._compute_statistics_list(data_first_sol)
            except:
                pass
        else:
            print(key, "not in data dict")
            
        
    def _compute_statistics_second_sol(self, key:str):
        if key in self.data_perfs.keys():
            data = self.data_perfs[key]
            
            try:
                print("---", key, "second solution found")
                data_second_sol = [sol[1] for sol in data if len(sol) > 1]
                self._compute_statistics_list(data_second_sol)
            except:
                pass
        else:
            print(key, "not in data dict")
            
    def compute_statistics_first_sol(self):
        for k in ["time_first", "n_nmpc_first", "it_first"]:
            self._compute_statistics_first_sol(k)
            self._compute_statistics_second_sol(k)
    
    def compute_statistics_contact_plans(self):
        data_contact_plan = self.data_perfs["contact_plan"]

        all_contact_plan = []
        for contact_plan_goal in data_contact_plan:
            for cp in contact_plan_goal:
                all_contact_plan.append(cp)
    
        print("---", "contact plan length")
        print("Total number of contact plans found", len(all_contact_plan))
        print("Average nb of contact plans found", len(all_contact_plan) / len(data_contact_plan))
        print("Total number of jumps", len([jump for jump in cp for cp in all_contact_plan]))
        
        len_data_contact_plan = [len(cp) - 2 for cp in all_contact_plan] # - 2 for repetition
        self._compute_statistics_list(len_data_contact_plan)
        
    def get_longest_contact_plans(self, N : int = 3):
        data_contact_plan = self.data_perfs["contact_plan"]

        all_contact_plan = []
        cp_length = []
        exp_env_goal_i_cp = []
        for path, contact_plans in zip(self.perfs.get_all_run_dir(self.exp_dir), data_contact_plan):
            for i_cp, cp in enumerate(contact_plans):
                all_contact_plan.append(cp)
                cp_length.append(len(cp))
                
                # Find i_env, i_goal from path
                l = path.split("/")
                env_str = l[-2]
                goal_str = l[-1]
                i_env = int(env_str.split("_")[-1])
                i_goal = int(goal_str.split("_")[-1])
                exp_env_goal_i_cp.append((i_env, i_goal, i_cp))
                    
        # Sort contact plans by length
        sorted_id = np.argsort(cp_length)[::-1]
        all_contact_plan_sorted = [all_contact_plan[i] for i in sorted_id]
        exp_env_goal_i_cp_sorted = [exp_env_goal_i_cp[i] for i in sorted_id]
        cp_length_sorted = [cp_length[i] for i in sorted_id]

        for i in range(N):
            print("---------")
            print(exp_env_goal_i_cp_sorted[i])
            print(cp_length_sorted[i])

    def compute_success(self):
        data_success = self.data_perfs["success"]
        
        print("---", "success")
        # Success rate
        success_rate = sum(data_success) / len(data_success) * 100
        print(sum(data_success), "success out of", len(data_success), "trials")
        print("Success rate", round(success_rate, 1), "%")
        

def main(exp_dir : str):
    perf_evaluation = MCTSPerformancesEvaluation(exp_dir)

if __name__ == "__main__":
    args = tyro.cli(main)
    
