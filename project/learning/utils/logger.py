from torch.utils.tensorboard import SummaryWriter

class LoggerAbstract():
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

    def write_scalar(self, name, value, epoch):
        pass

    def update_layout(self, add_layout:dict) -> None:
        pass


class TensorBoardLogger(LoggerAbstract):
    def __init__(self,
                 log_folder:str,
                 comment:str=""
                 ) -> None:
        super().__init__(log_folder)
        self.comment = comment
        self.writer = SummaryWriter(self.log_folder, self.comment)

        self.layout = {
            "Metrics": {
                "loss": ["Multiline", ["loss/train", "loss/val"]],
                "accuracy": ["Multiline", ["accuracy/train", "accuracy/val"]],
            },
            "Utils" : {
                "LR":  ["Multiline", ["LR/values"]],
            }
        }
        self.custom_scalars = False

    def _custom_scalars(self):
        if not self.custom_scalars:
            self.writer.add_custom_scalars(self.layout)
            self.custom_scalars = True

    def update_layout(self, add_layout:dict) -> None:
        self.layout.update(add_layout)

    def write_hparams(self, cfg:dict, metric:dict={}):
        self._custom_scalars()
        self.writer.add_hparams(cfg, metric_dict=metric)

    def write_scalar(self, name, value, epoch):
        self._custom_scalars()
        self.writer.add_scalar(name, value, global_step=epoch)

    def write_scalars(self, name, loss_dict, epoch):
        self.writer.add_scalars(name, loss_dict, global_step=epoch)

    def write_figure(self, name, fig, epoch):
        self.writer.add_figure(tag=name, figure=fig, global_step=epoch)

    def close(self):
        self.writer.close()