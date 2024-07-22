import tyro

from utils.replay import ExperimentReplay

def main(exp_dir : str,
         env : int = 0,
         goal : int = 0,
         cp : int = 0,
         viewer : bool = True,
         record_video : bool = True,
         video_subdir : str = "",
         controller : str = "biconmp"
         ):
    
    replay = ExperimentReplay(exp_dir)
    
    replay.run(
        controller=controller,
        i_env=env,
        i_goal=goal,
        i_cp=cp,
        use_viewer=viewer,
        record_video=record_video,
        video_subdir=video_subdir
    )
        
if __name__ == "__main__":
    args = tyro.cli(main)