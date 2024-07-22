from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self,
                 log_folder:str
                 ) -> None:
        self.log_folder = log_folder

    def write_scalars(self, name, loss_dict, epoch):
        pass

    def write_hparams(self, cfg:dict, metric:dict={}):
        pass

    def close(self):
        pass

class TensorBoardLogger(Logger):
    def __init__(self,
                 log_folder:str,
                 comment:str=""
                 ) -> None:
        super().__init__(log_folder)
        self.comment = comment
        self.writer = SummaryWriter(self.log_folder, self.comment)

    def write_hparams(self, cfg:dict, metric:dict={}):
        self.writer.add_hparams(cfg, metric_dict=metric)

    def write_scalars(self, name, loss_dict, epoch):
        self.writer.add_scalars(name, loss_dict, global_step=epoch)

    def close(self):
        self.writer.close()