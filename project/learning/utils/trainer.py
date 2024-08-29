

import os, glob
import torch
import math
import inspect
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from .config import Config
from .logger import LoggerAbstract, TensorBoardLogger
from .log_manager import LogManager
from .diffusion.DDPM import DDPM
from .diffusion.CDCD import CDCD
from .diffusion.visualization import get_noise_schedule_figures

class TrainerBase():
    """
    Trainer base class for training PyTorch models.
    """

    DEFAULT_BACTH_SIZE = 32
    DEFAULT_OPTIMIZER = {"optimizer_name" : "Adam"}
    DEFAULT_CRITERION = "MSELoss"
    DEFAULT_EPOCHS = 100
    DEFAULT_LOGDIR = "./logs"
    DEFAULT_DIR_NAME = ""
    DEFAULT_SCHEDULER = {}
    DEFAULT_TRAIN_TEST_RATIO = 5
    DEFAULT_USE_LOGGER = True
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEFAULT_PLOT_FIGURES = []

    def __init__(self,
                 cfg:Config,
                 model:Module,
                 dataloader_train:DataLoader,
                 dataloader_test:DataLoader=None,
                 **kwargs) -> None:
        """
        Initialize Trainer.

        Args:
            - cfg (Config): Configuration object.
            - model (Module): PyTorch model.
            - dataloader_train (DataLoader): Training data loader.
            - dataloader_test (Optional[DataLoader]): Testing data loader (default: None).
            - **kwargs: Additional optional parameters.
        """
        self.cfg = cfg
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.model = model

        self.current_epoch = 0
        self.logs_metrics_dict = {}
        self.logs_utils_dict = {}
        self.metrics_dict = {}
        self.train_loss_history = []
        self.val_loss_history = []
        self.min_val_loss = math.inf
        
        # Optional arguments
        self.optional_args = {
            "batch_size" : TrainerBase.DEFAULT_BACTH_SIZE,
            "epochs" : TrainerBase.DEFAULT_EPOCHS,
            "optimizer" : TrainerBase.DEFAULT_OPTIMIZER,
            "criterion_str" : TrainerBase.DEFAULT_CRITERION,
            "lr_scheduler" : TrainerBase.DEFAULT_SCHEDULER,
            "logdir" : TrainerBase.DEFAULT_LOGDIR,
            "train_test_ratio" : TrainerBase.DEFAULT_TRAIN_TEST_RATIO,
            "use_logger" : TrainerBase.DEFAULT_USE_LOGGER,
            "device" : TrainerBase.DEFAULT_DEVICE,
            "plot_figures" : TrainerBase.DEFAULT_PLOT_FIGURES,
            "run_dir_name" : TrainerBase.DEFAULT_DIR_NAME,
        }

        self.optional_args.update(kwargs)

        self._set_config_params()
        self._set_optimizer()
        self._set_lr_scheduler()
        self._set_criterion()

        # Log manager
        self.log_manager = LogManager(self.logdir, self.run_dir_name)
        self.run_dir = self.log_manager.run_dir
        self.model_name = self.model.name if hasattr(self.model, "name") else "model"
        self.model_path = os.path.join(self.run_dir, f"{self.model_name}.pth")
        
        # Load model
        self.model = model.to(self.device)
        if os.path.exists(self.model_path):
            self._load_state(self.model_path)
        else:
            self._copy_config(cfg.config_path)
                
        # Logger
        self.logger = LoggerAbstract(self.run_dir)
        if self.use_logger and not(self.log_manager._is_run_dir(self.run_dir)):
            self.logger = TensorBoardLogger(self.run_dir, kwargs.get("comment", ""))

    def _set_config_params(self) -> None:
        """
        Update default parameters with configuration values.
        """
        for k, v in self.optional_args.copy().items():
            cfg_value = self.cfg.get_value(k)
            if cfg_value != None:
                v = cfg_value
                self.optional_args[k] = v
            setattr(self, k, v)

        for k, v in self.cfg.training.items():
            self.optional_args[k] = v
            setattr(self, k, v)
        
    def _set_optimizer(self):
        """
        Set the optimizer for training.
        """
        self.optim = None
        if isinstance(self.optimizer, dict) and len(self.optimizer) > 0:
            optimizer_name = self.optimizer.get("optimizer_name")
            optimizer_params = self.optimizer.get("PARAMS", {})
            if optimizer_name:
                optimizer_cls = getattr(optim, optimizer_name, None)
                if optimizer_cls is not None:
                    self.optim = optimizer_cls(self.model.parameters(), **optimizer_params)
                else:
                    print(f"Optimizer '{optimizer_name}' not found.")

    def _set_lr_scheduler(self):
        """
        Set the learning rate scheduler.
        """
        self.scheduler = None
        if isinstance(self.lr_scheduler, dict) and len(self.lr_scheduler) > 0:
            lr_scheduler_name = self.lr_scheduler.get("lr_scheduler_name")
            lr_scheduler_params = self.lr_scheduler.get("PARAMS", {})

            if lr_scheduler_name:
                scheduler_cls = getattr(lr_scheduler, lr_scheduler_name, None)
                if scheduler_cls is not None:
                    self.scheduler = scheduler_cls(self.optim, **lr_scheduler_params)
                else:
                    print(f"Lr scheduler '{lr_scheduler_name}' not found.")

    def _set_criterion(self):
        """
        Set the loss criterion.
        """
        criterion_cls = getattr(nn, self.criterion_str, None)
        if criterion_cls is not None:
            self.criterion = criterion_cls()
        else:
            print(f"Loss '{self.criterion_str}' not found.")

    def _save_model(self) -> None:
        """
        Save model and optimizer state to run_dir.
        """
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'train_loss_history': self.train_loss_history,
            'val_loss_history': self.val_loss_history,
            'trainer' : self.__class__.__name__,
        }
        torch.save(state, self.model_path)

    def _load_state(self, state_path:str) -> None:
        """
        Load the state of a pretrained model.

        Args:
            - state_path (str): Path to the model state.
        """
        # If path is run dir
        if state_path[-4:] != ".pth" and self.log_manager._is_run_dir(state_path):
            pth_paths = glob.glob(os.path.split(state_path)[0] + "/*.pth")
            if len(pth_paths) > 0:
                state_path = pth_paths[0]
            else:
                print(f"No .pth in directory {state_path}.")
                return None
        try:
            state = torch.load(state_path)
            self.current_epoch = state["epoch"]
            self.model.load_state_dict(state["state_dict"])
            self.optim.load_state_dict(state["optimizer"])
            self.train_loss_history = state["train_loss_history"]
            self.val_loss_history = state["val_loss_history"]

            print(f"Resuming training at {self.run_dir}, epoch {self.current_epoch}")

        except Exception as e:
            print(e)
            print(f"Can't load state at {state_path}.")

    def _copy_config(self, config_path):
        """
        Copy the configuration file to the run directory.

        Args:
            - config_path: Path to the configuration file.
        """
        file_name = os.path.split(config_path)[-1]
        path = os.path.join(self.log_manager.run_dir, file_name)
        self.cfg.write(path)
        
    def _write_results(self):
        """
        Write training results to a file.
        """
        with open(os.path.join(self.run_dir, "results.txt"), "w") as file:
            file.write(f"Train loss: {self.train_loss_history[-1]}\n")
            if len(self.val_loss_history) > 0:
                file.write(f"Test loss: {self.val_loss_history[-1]}")

    def _save_training_curves(self):
        """
        Save training curves plot.
        """
        fig, ax = plt.subplots()
        ax.plot(self.train_loss_history, label = "train")
        if len(self.val_loss_history) > 0:
            ax.plot(self.val_loss_history, label = "test")
        plt.yscale("log")
        plt.legend()
        plt.title("Loss history")
        plt.savefig(os.path.join(self.run_dir, "loss_history.png"))

    def _write_logs(self):
        """
        Write logs to the logger.
        """
        for name, loss in self.logs_metrics_dict.items():
            self.logger.write_scalar(name, loss, self.current_epoch)

        for name, value in self.logs_utils_dict.items():
            self.logger.write_scalar(name, value, self.current_epoch)

    def _write_hparams(self):
        """
        Write hyperparameters to the logger.
        """
        self.logger.write_hparams(self.cfg.get_cfg_as_dict(), self.metrics_dict)

    def _update_lr(self):
        """
        Update learning rate according to scheduler
        """
        if self.scheduler:
            self.scheduler.step()
            current_lr = self.scheduler.get_lr()
            self.logger.write_scalar("LR/values", current_lr, self.current_epoch)


    def _write_figures(self, batch):
        """
        Save the plots in logs writer.

        Execute all functions in self.plot_figures.
        Those should be function that takes in input:
            - A data batch
            - A model
        and returns:
            - plt.fig or a list of plt.fig
        """
        if self.logger:
            for i, plot_fn in enumerate(self.plot_figures):
                fig_list = plot_fn(batch, self.model)

                if not isinstance(fig_list, list):
                    fig_list = [fig_list]
                
                for fig in fig_list:

                    title = fig.axes[0].get_title()

                    if title == "":
                        title = f"figure_{i}"

                    self.logger.write_figure(title, fig, self.current_epoch)
        

####################################################
###################  SUPERVISED  ###################
####################################################

class TrainerSupervised(TrainerBase):
    """
    Trainer specialized for supervised learning tasks.
    """
    def __init__(self,
                 cfg:Config,
                 model:Module,
                 dataloader_train:DataLoader,
                 dataloader_test:DataLoader=None,
                 **kwargs) -> None:
        """
        Trainer specialized for supervised learning tasks.

        Args:
            - cfg (Config): Configuration object.
            - model (Module): PyTorch model.
            - dataloader_train (DataLoader): Training data loader.
            - dataloader_test (Optional[DataLoader]): Testing data loader (default: None).
            - **kwargs: Additional optional parameters.
        """
        super().__init__(cfg,
                         model,
                         dataloader_train,
                         dataloader_test,
                         **kwargs)

    def _train_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            - float: Average training loss for the epoch.
        """
        total_loss = 0.

        self.model.train()
        for batch in tqdm(self.dataloader_test, desc = "Batch", leave=False):
            input = batch.pop("input")
            target = batch.pop("target", None)
            # Move batch to device
            input = input.to(self.device)
            target = target.to(self.device)

            out = self.model(input)
            loss = self.criterion(out, target)
            loss.backward()
            self.optim.zero_grad()

            total_loss += loss.item()

        # Update learning rate
        self._update_lr()
        
        # Calculate average loss for the epoch
        train_loss = total_loss / len(self.dataloader_train)

        # Write logs and history
        self.train_loss_history.append(train_loss)
        self.logs_metrics_dict["loss/train"] = train_loss
        
        return train_loss

    @torch.no_grad()
    def _test_epoch(self):
        """
        Evaluate the model on the testing dataset for one epoch.

        Returns:
            - float: Average testing loss for the epoch.
        """
        total_loss = 0.

        self.model.eval()
        for batch in tqdm(self.dataloader_test, desc = "Batch", leave=False):
            input = batch.pop("input")
            target = batch.pop("target")
            # Move batch to device
            input = input.to(self.device)
            target = target.to(self.device)

            out = self.model(input)
            loss = self.criterion(out, target)
            total_loss += loss.item()

        # Calculate average loss for the epoch
        val_loss = total_loss / len(self.dataloader_test)

        # Write logs and history
        self.val_loss_history += [val_loss] * self.train_test_ratio
        self.logs_metrics_dict["loss/val"] = val_loss

        return val_loss

    def train(self):
        """
        Train the model for multiple epochs.
        """
        progress_bar = trange(self.epochs, desc = "Epochs")

        for epoch in progress_bar:

            # Train
            self._train_epoch()

            # Test
            if (self.dataloader_test != None and
                epoch > 0 and
                (epoch + 1) % self.train_test_ratio == 0):

                val_loss = self._test_epoch()
                # Save model with best test loss
                if (epoch != 0 and val_loss < self.min_val_loss):
                    self._save_model()
                    self.min_val_loss = val_loss
                # Else, save model every <ckpt_every>
            else:
                if (epoch == self.epochs - 1 or 
                   (epoch % self.ckpt_every and epoch > 0)):
                    self._save_model()

            # Write logs
            self._write_logs()

            progress_bar.set_postfix(self.logs_metrics_dict)
            self.current_epoch += 1
        
        self.metrics_dict["hparam/min_val_loss"] = self.min_val_loss

        self._write_hparams()
        self._write_results()
        self._save_training_curves()

####################################################
####################    DDPM    ####################
####################################################
        
class TrainerDDPM(TrainerSupervised):
    """
    Trainer specialized for DDPM (Denosing Diffusion Probabilistic Models).
    """
    DEFAULT_EMA_POWER = 0.
    DEFAULT_CKPT_EVERY = 1
    DEFAULT_WARM_START_SNR = 20 # Train model first for 20 epochs

    def __init__(self,
                 cfg:Config,
                 model:Module,
                 dataloader_train:DataLoader,
                 dataloader_test:DataLoader,
                 **kwargs) -> None:
        """
        Trainer class for training PyTorch models.

        Args:
            - cfg (Config): Configuration object.
            - model (Module): PyTorch model.
            - dataloader_train (DataLoader): Training data loader.
            - dataloader_test (Optional[DataLoader]): Testing data loader (default: None).
            - **kwargs: Additional optional parameters.
        """

        # Optional arguments
        optional_args = {
            "ema_power" : TrainerDDPM.DEFAULT_EMA_POWER,
            "ckpt_every" : TrainerDDPM.DEFAULT_CKPT_EVERY,
        }
        optional_args.update(kwargs)

        if not isinstance(model, DDPM) or not isinstance(model, CDCD):
            ddpm_model = DDPM(model, **cfg.model["PARAMS"])
        else:
            ddpm_model = model

        super().__init__(cfg,
                         ddpm_model,
                         dataloader_train,
                         dataloader_test,
                         **optional_args)

        self.ema = EMAModel(
            parameters=self.model.parameters(),
            power=self.ema_power) if self.ema_power > 0. else None

        # Custom hugging face scheduler
        lr_scheduler_str = self.lr_scheduler.pop("lr_scheduler_name", None)
        if lr_scheduler_str and not self.scheduler:
            self.scheduler = get_scheduler(
                name=lr_scheduler_str,
                optimizer=self.optim,
                num_warmup_steps=min(self.cfg.get_value("epochs") / 20, 500),
                num_training_steps= len(self.dataloader_train) * self.cfg.get_value("epochs")
            )

        if self.model.learn_snr:
            # Add plots
            self.plot_figures.append(get_noise_schedule_figures)

            # Track snr model params
            self.snr_model_logs_name = self.model.noise_scheduler.snr_model.__class__.__name__
            weight_model_layout = {
                "Utils" : {
                    "LR":  ["Multiline", ["LR/values"]],
                    self.snr_model_logs_name :  ["Multiline", []],
                }
            }

            for name, _ in self.model.noise_scheduler.snr_model.named_parameters():
                logs_name = f"{self.snr_model_logs_name}/{name}"
                weight_model_layout["Utils"][self.snr_model_logs_name][1].append(logs_name)
            
            self.logger.update_layout(weight_model_layout)

    def _write_weights_snr_model(self):
        if self.model.learn_snr:
            for name, value in self.model.noise_scheduler.snr_model.named_parameters():
                logs_name = f"{self.snr_model_logs_name}/{name}"
                if len(value) > 0:
                    value = torch.mean(value)
                self.logger.write_scalar(logs_name, value, self.current_epoch)

    def _update_ema(self):
        """
        Update weights with exponential moving average.
        """
        if self.ema:
            self.ema.step(self.model.parameters())

    def _train_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            - float: Average loss for the epoch.
        """
        total_loss = 0.

        self.model.train()
        for batch in tqdm(self.dataloader_train, desc = "Batch", leave=False):
            data_samples = batch.pop("data")
            condition = batch.pop("condition", None)

            data_samples = data_samples.to(self.device)
            condition = condition.to(self.device) if condition != None else None

            loss = self.model(data_samples, None, condition, criterion=self.criterion)
            total_loss += loss.item()

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            
            # Update Exponential Moving Average of the model weights
            self._update_ema()

        # Update learning rate
        self._update_lr()

        # Write snr model weights
        self._write_weights_snr_model()

        # Calculate average loss for the epoch
        train_loss = total_loss / len(self.dataloader_train)

        # Write logs and history
        self.train_loss_history.append(train_loss)
        self.logs_metrics_dict["loss/train"] = train_loss
        
        return train_loss

    @torch.no_grad()
    def _test_epoch(self):
        """
        Evaluate the model on the testing dataset for one epoch.

        Returns:
            - float: Average testing loss for the epoch.
        """
        total_loss = 0.

        self.model.eval()
        for batch in tqdm(self.dataloader_test, desc = "Batch", leave=False):
            data_samples = batch.pop("data")
            condition = batch.pop("condition", None)

            data_samples = data_samples.to(self.device)
            condition = condition.to(self.device) if condition != None else None

            data_generated = self.model.sample(condition=condition)
            loss = torch.nn.functional.mse_loss(data_generated, data_samples)
            total_loss += loss.item()

        # Calculate average loss for the epoch
        val_loss = total_loss / len(self.dataloader_test)

        # Write logs and history
        self.val_loss_history += [val_loss] * self.train_test_ratio
        self.logs_metrics_dict["loss/val"] = val_loss

        # Write figures
        self._write_figures(batch)

        return val_loss

####################################################
####################    CDCD    ####################
####################################################
        
class TrainerCDCD(TrainerDDPM):
    """
    Trainer specialized for CDCD (Continuous Diffusion with Categorical Data).
    """

    def __init__(self,
                 cfg:Config,
                 model:Module,
                 dataloader_train:DataLoader,
                 dataloader_test:DataLoader,
                 **kwargs) -> None:
        """
        Trainer class for training PyTorch models.

        Args:
            - cfg (Config): Configuration object.
            - model (Module): PyTorch model.
            - dataloader_train (DataLoader): Training data loader.
            - dataloader_test (Optional[DataLoader]): Testing data loader (default: None).
            - **kwargs: Additional optional parameters.
        """

        # Optional arguments
        optional_args = {}
        optional_args.update(kwargs)

        if not isinstance(model, CDCD):
            cdcd_model = CDCD(model, **cfg.model["PARAMS"])
        else:
            cdcd_model = model

        super().__init__(cfg,
                         cdcd_model,
                         dataloader_train,
                         dataloader_test,
                         **optional_args)
        
    def _train_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            - float: Average loss for the epoch.
        """
        total_loss = 0.

        self.model.train()
        for batch in tqdm(self.dataloader_train, desc = "Batch", leave=False):
            data_samples = batch.pop("data")
            condition = batch.pop("condition", None)
            index = batch.pop("index", None)

            data_samples = data_samples.to(self.device)
            condition = condition.to(self.device) if condition != None else None
            index = index.to(self.device) if index != None else None

            loss = self.model(data_samples, index, None, condition, criterion=self.criterion)
            total_loss += loss.item()

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            
            # Update Exponential Moving Average of the model weights
            self._update_ema()

        # Update learning rate
        self._update_lr()

        # Write snr model weights
        self._write_weights_snr_model()

        # Calculate average loss for the epoch
        train_loss = total_loss / len(self.dataloader_train)

        # Write logs and history
        self.train_loss_history.append(train_loss)
        self.logs_metrics_dict["loss/train"] = train_loss
        
        return train_loss

    @torch.no_grad()
    def _test_epoch(self):
        """
        Evaluate the model on the testing dataset for one epoch.

        Returns:
            - float: Average testing loss for the epoch.
        """
        total_loss = 0.

        self.model.eval()
        for batch in tqdm(self.dataloader_test, desc = "Batch", leave=False):
            data_samples = batch.pop("data")
            condition = batch.pop("condition", None)

            data_samples = data_samples.to(self.device)
            condition = condition.to(self.device) if condition != None else None

            data_generated = self.model.sample(condition=condition)
            loss = torch.nn.functional.mse_loss(data_generated, data_samples)
            total_loss += loss.item()

        # Calculate average loss for the epoch
        val_loss = total_loss / len(self.dataloader_test)

        # Write logs and history
        self.val_loss_history += [val_loss] * self.train_test_ratio
        self.logs_metrics_dict["loss/val"] = val_loss

        # Write figures
        self._write_figures(batch)

        return val_loss

class TrainerFactory():
    TRAINER = {
        "supervised": TrainerSupervised,
        "ddpm": TrainerDDPM,
        "cdcd": TrainerCDCD,
    }

    @staticmethod
    def create_trainer(training_mode:str):
        trainer_class = TrainerFactory.TRAINER.get(training_mode.lower())
        if not trainer_class:
            raise ValueError("Invalid training mode. Should be supervised or ddpm.")
        return trainer_class

