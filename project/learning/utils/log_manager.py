import os
import glob 

class LogManager():
    def __init__(self, logdir:str, run_dir_name:str="") -> None:
        self.logdir = logdir
        self.run_dir_name = run_dir_name
        self._create_if_not_exists(self.logdir)

        self.run_dir = ""
        if (self._is_run_dir(self.logdir)):
            self.run_dir = self.logdir
        else:
            self._create_next_run_folder()

        print("Run directory", self.run_dir)

    def _create_if_not_exists(self, path):
        if not(os.path.exists(path)):
            os.makedirs(path)

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

        if not(os.path.exists(self.run_dir)):
            os.mkdir(self.run_dir)

