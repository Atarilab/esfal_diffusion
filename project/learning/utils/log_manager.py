import os
import glob
import shutil 

from .model_utils import DEFAULT_MODEL_PATH, DEFAULT_MODELS_DIR

class LogManager():
    def __init__(self, logdir:str, run_dir_name:str="") -> None:
        self.logdir = logdir
        self.run_dir_name = run_dir_name
        os.makedirs(self.logdir, exist_ok=True)

        # Create a new logdir if not a existing one
        self.run_dir = ""
        if (self._is_run_dir(self.logdir)):
            self.run_dir = self.logdir
        else:
            self._create_next_run_folder()
            self._copy_models_to_run_dir()
        
        print("Run directory", self.run_dir)

    def _is_run_dir(self, path):
        """
        Return True is <path> is a log run directory 
        that contains config and a model
        """
        rundir = False
        if (len(glob.glob(path + "/*.yml")) > 0 or \
            len(glob.glob(path + "/*.yaml")) > 0) and \
            len(glob.glob(path + "/*.pth")) > 0:
            rundir = True
        
        return rundir

    def _get_max_log_id(self):
        """
        Return max log id in log directory.
        """
        max_log_id = -1
        for log_path in os.listdir(self.logdir):
            log_id = None
            try:
                log_id = int(log_path.split("_")[0])
            except:
                pass

            if log_id != None and log_id > max_log_id:
                max_log_id = log_id

        return max_log_id
    
    def _create_next_run_folder(self):
        """
        Create a new run folder.
        """
        log_id = self._get_max_log_id() + 1
        dir_name = f"{log_id}_{self.run_dir_name}" if self.run_dir_name != "" else str(log_id)
        self.run_dir = os.path.join(self.logdir, dir_name)
        os.makedirs(self.run_dir, exist_ok=True)

    def _copy_models_to_run_dir(self):
        """
        Copy model files to run dir.
        Model loader will use that version in the model when loading the model state.
        """
        
        # Copy models files to run directory
        run_models_path = os.path.join(self.run_dir, DEFAULT_MODELS_DIR)
        if not os.path.exists(run_models_path):
            shutil.copytree(DEFAULT_MODEL_PATH, run_models_path)

    def remove_run_dir(self):
        """
        Remove run dir.
        """
        shutil.rmtree(self.run_dir, ignore_errors=True)