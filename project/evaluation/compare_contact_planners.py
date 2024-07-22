import os

from utils.eval import RerunExperiments
from utils.eval import PerformanceGatherer

class RerunEvaluation():
    def __init__(self, rerun_dir : str) -> None:
        self.rerun_dir = rerun_dir
        
        self.compute_success_rate()
        self.compute_success_rate_per_contact_plan_length()
        
    def compute_success_rate(self):
        data_success = []

        for goal_dir in PerformanceGatherer.get_all_run_dir(self.rerun_dir):
            success = len(os.listdir(goal_dir)) >  0
            data_success.append(int(success))
                    
        print("---", "success")
        # Success rate
        success_rate = sum(data_success) / len(data_success) * 100
        print(sum(data_success), "success out of", len(data_success), "trials")
        print("Success rate", round(success_rate, 1), "%")
        
    def compute_success_rate_per_contact_plan_length(self):
        data_success_cp_length = {}
        data_success_achieved = []
        run_dir = os.path.split(self.rerun_dir)[0]
        run_perfs = PerformanceGatherer(run_dir)
        
        i_env = 0
        i_goal = 0
        for goal_dir in PerformanceGatherer.get_all_run_dir(self.rerun_dir):
            success = len(os.listdir(goal_dir)) >  0
            # Compute contact plan min length
            data_run = run_perfs.get_data(i_env, i_goal)
            min_length_cp = 0
            if len(data_run) > 0:
                all_contact_plans = data_run["contact_plan"]
                if len(all_contact_plans) > 0:
                    min_length_cp = min([len(cp) - 2 for cp in all_contact_plans])
                    data_success_achieved.append(success)
                    
            if min_length_cp not in data_success_cp_length.keys():
                data_success_cp_length[min_length_cp] = []
                
            
            data_success_cp_length[min_length_cp].append(success)
            
            i_env += 1
            
            
        print("---", "success per cp length")
        
        # Success rate
        for cp_length, data_success in data_success_cp_length.items():
            success_rate = sum(data_success) / len(data_success) * 100
            print("Number of jumps", cp_length)
            print(sum(data_success), "success out of", len(data_success), "trials")
            print("Success rate", round(success_rate, 1), "%")
            
        print("---", "success MCTS achieved")
        # Success rate
        success_rate = sum(data_success_achieved) / len(data_success_achieved) * 100
        print(sum(data_success_achieved), "success out of", len(data_success_achieved), "trials")
        print("Success rate", round(success_rate, 1), "%")
            
            

if __name__ == "__main__":
    # experiment_dir = "/home/atari_ws/data/trot/"
    # rerun = RerunExperiments(experiment_dir, contact_planner="raibert")
    # rerun.start(15)
    
    rerun_experiment_dir = "/home/atari_ws/data/trot/_rerun_raibert_1307-131411"
    evaluation = RerunEvaluation(rerun_experiment_dir)
