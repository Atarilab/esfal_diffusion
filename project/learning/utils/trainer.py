import os, glob
import torch
import math
import inspect
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.nn import Module
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from .config import Config
from .logger import Logger, TensorBoardLogger
from .log_manager import LogManager

DEFAULT_BACTH_SIZE = 32
DEFAULT_OPTIMIZER = {"optimizer_name" : "Adam"}
DEFAULT_CRITERION = "MSELoss"
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 100
DEFAULT_LOGDIR = "./logs"
DEFAULT_SCHEDULER = {}
DEFAULT_DATASET = ""
DEFAULT_TRAIN_TEST_RATIO = 5
DEFAULT_USE_LOGGER = True
DEFAULT_MODEL_STATE = ""
DEFAULT_GRADIENT_CLIPPING = 0.

class TrainerBase():
    """
    Trainer base class for training PyTorch models.
    """
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test

        self.current_epoch = 0
        self.logs_loss_dict = {}
        self.metrics_dict = {}
        self.train_loss_history = []
        self.test_loss_history = []
        self.min_test_loss = math.inf
        self.nargs_model = len(inspect.signature(self.model.forward).parameters)
        
        # Optional arguments
        self.optional_args = {
            **kwargs, 
            **{
            "dataset" : DEFAULT_DATASET,
            "batch_size" : DEFAULT_BACTH_SIZE,
            "epochs" : DEFAULT_EPOCHS,
            "optimizer" : DEFAULT_OPTIMIZER,
            "lr" : DEFAULT_LR,
            "criterion_str" : DEFAULT_CRITERION,
            "lr_scheduler" : DEFAULT_SCHEDULER,
            "logdir" : DEFAULT_LOGDIR,
            "train_test_ratio" : DEFAULT_TRAIN_TEST_RATIO,
            "use_logger" : DEFAULT_USE_LOGGER,
            "model_state" : DEFAULT_MODEL_STATE,
            "gradient_clipping" : DEFAULT_GRADIENT_CLIPPING,
            }
        }
        
        self._set_config_params()
        self._set_optimizer()
        self._set_lr_scheduler()
        self._set_criterion()

        # Log manager
        self.log_manager = LogManager(self.logdir)
        self.run_dir = self.log_manager.run_dir
        self.model_name = self.model.name if hasattr(self.model, "name") else "model"
        self.model_path = os.path.join(self.run_dir, f"{self.model_name}.pth")
        if os.path.exists(self.model_path):
            self.load_state(self.model_path)

        # Load model
        if self.model_state != "":
            self.load_state(self.model_state)
        else:
            self.copy_config(cfg.config_path)
                
        # Logger
        self.logger = Logger(self.run_dir)
        if self.use_logger and not(self.log_manager._is_run_dir(self.run_dir)):
            self.logger = TensorBoardLogger(self.run_dir, kwargs.get("comment", ""))

    def _set_config_params(self) -> None:
        """
        Update default parameters with configuration values.
        """
        for k, v in self.optional_args.copy().items():
            cfg_value = self.cfg.get_value(k)
            if cfg_value != None:
                self.optional_args[k] = cfg_value
                v = cfg_value
            setattr(self, k, v)
        
    def _set_optimizer(self):
        """
        Set the optimizer for training.
        """
        if type(self.optimizer) == dict and len(self.optimizer) > 0:
            try:
                optimizer_str = self.optimizer["optimizer_name"]
                optimizer_params = self.optimizer.get("PARAMS", {})
                exec(f"setattr(self, 'optimizer', torch.optim.{optimizer_str}(self.model.parameters(), **optimizer_params))")
            except AttributeError as e:
                print(e)
                print(f"Optimizer '{self.optimizer_str}' not found.")


    def _set_lr_scheduler(self):
        """
        Set the learning rate scheduler.
        """
        self.scheduler = None
        if type(self.lr_scheduler) == dict and len(self.lr_scheduler) > 0:
            try:
                lr_scheduler_str = self.lr_scheduler["lr_scheduler_name"]
                lr_scheduler_params = self.lr_scheduler["PARAMS"]
                exec(f"setattr(self, 'scheduler', torch.optim.lr_scheduler.{lr_scheduler_str}(self.optimizer, **lr_scheduler_params))")
            except ValueError as e:
                print(e)
                print(f"Lr scheduler '{self.lr_scheduler}' not found.")
    
    def _set_criterion(self):
        """
        Set the loss criterion.
        """
        try:
            exec(f"setattr(self, 'criterion', torch.nn.{self.criterion_str}())")
        except ValueError as e:
            print(e)
            print(f"Loss '{self.criterion_str}' not found.")

    def _save_model(self) -> None:
        """
        Save model and optimizer state to run_dir.
        """
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss_history': self.train_loss_history,
            'test_loss_history': self.test_loss_history
        }
        torch.save(state, self.model_path)

    def load_state(self, state_path:str) -> None:
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
            self.optimizer.load_state_dict(state["optimizer"])
            self.train_loss_history = state["train_loss_history"]
            self.test_loss_history = state["test_loss_history"]

            print(f"Resuming training at {self.run_dir}, epoch {self.current_epoch}")

        except Exception as e:
            print(e)
            print(f"Can't load state at {state_path}.")


    def copy_config(self, config_path):
        """
        Copy the configuration file to the run directory.

        Args:
            - config_path: Path to the configuration file.
        """
        file_name = os.path.split(config_path)[-1]
        path = os.path.join(self.log_manager.run_dir, file_name)
        self.cfg.write(path)
        
    def write_results(self):
        """
        Write training results to a file.
        """
        with open(os.path.join(self.run_dir, "results.txt"), "w") as file:
            file.write(f"Train loss: {self.train_loss_history[-1]}\n")
            if len(self.test_loss_history) > 0:
                file.write(f"Test loss: {self.test_loss_history[-1]}")

    def save_training_curves(self):
        """
        Save training curves plot.
        """
        fig, ax = plt.subplots()
        ax.plot(self.train_loss_history, label = "train")
        if len(self.test_loss_history) > 0:
            ax.plot(self.test_loss_history, label = "test")
        plt.yscale("log")
        plt.legend()
        plt.title("Loss history")
        plt.savefig(os.path.join(self.run_dir, "loss_history.png"))

    def write_loss(self):
        """
        Write loss to the logger.
        """
        self.logger.write_scalars("Loss", self.logs_loss_dict, self.current_epoch)

    def write_hparams(self, cfg:dict):
        """
        Write hyperparameters to the logger.

        Args:
            - cfg (dict): Dictionary of hyperparameters.
        """
        self.logger.write_hparams(cfg, self.metrics_dict)

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
        for batch in tqdm(self.dataloader_train, desc = "Batch", leave=False):
            # Move batch to device
            input = batch["input"].to(self.device)
            target = batch["target"].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(input)
            loss = self.criterion(out, target)
            loss.backward()
            if self.gradient_clipping > 0.:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            self.optimizer.step()

            total_loss += loss.item()
             
        # Calculate average loss for the epoch
        average_loss = total_loss / len(self.dataloader_train)        
        
        return average_loss

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
            # Move batch to device
            input = batch["input"].to(self.device)
            target = batch["target"].to(self.device)
            
            if hasattr(self.model, "pointers"):
                out, probs = self.model.select(input)
                target_box = torch.gather(input, 1, target.unsqueeze(-1).expand(-1, -1, 3).long())
                loss = torch.nn.functional.mse_loss(out, target_box)
            else:
                out = self.model(input)
                loss = self.criterion(out, target)
            total_loss += loss.item()

        # Calculate average loss for the epoch
        average_loss = total_loss / len(self.dataloader_test)

        return average_loss

    def train(self):
        """
        Train the model for multiple epochs.
        """
        progress_bar = trange(self.epochs, desc = "Epochs")

        for epoch in progress_bar:

            # Train
            train_loss = self._train_epoch()

            # Test
            test_loss = None
            if (self.dataloader_test != None and epoch % self.train_test_ratio == 0):
                test_loss = self._test_epoch()
                # Save model with best test loss
                if (epoch != 0 and test_loss < self.min_test_loss):
                    self._save_model()
                    self.min_test_loss = test_loss
    
            # Write logs and history
                self.logs_loss_dict["tst_loss"] = test_loss

            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)

            self.logs_loss_dict["tr_loss"] = train_loss

            self.logger.write_scalars("Loss", self.logs_loss_dict, self.current_epoch)
            progress_bar.set_postfix(self.logs_loss_dict)

            self.current_epoch += 1
            if self.scheduler != None:
                self.scheduler.step()
        
        # Save last epoch model if only train
        if self.dataloader_test == None:
            self._save_model()
            self.metrics_dict["test_loss"] = torch.min(torch.tensor(self.train_loss_history)).item()
        else:
            self.metrics_dict["test_loss"] = self.min_test_loss
        self.write_hparams(self.cfg.get_cfg_as_dict())
        self.write_results()
        self.save_training_curves()

####################################################
####################    DDPM    ####################
####################################################
        
DEFAULT_EMA_POWER = 0.
DEFAULT_CKPT_EVERY = 1

class TrainerDDPM(TrainerBase):
    """
    Trainer specialized for DDPM (Denosing Diffusion Probabilistic Models).
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
        kwargs = {
            **kwargs,
            **{
                "ema_power" : DEFAULT_EMA_POWER,
                "ckpt_every" : DEFAULT_CKPT_EVERY,
            }
        }

        super().__init__(cfg,
                         model,
                         dataloader_train,
                         dataloader_test,
                         **kwargs)
        
        self.ema = EMAModel(
            parameters=self.model.parameters(),
            power=self.ema_power) if self.ema_power > 0. else None

        # Custom hugging face scheduler
        lr_scheduler_str = self.lr_scheduler.pop("lr_scheduler_name", None)
        if lr_scheduler_str and not self.scheduler:
            self.scheduler = get_scheduler(
                name=lr_scheduler_str,
                optimizer=self.optimizer,
                num_warmup_steps=min(self.cfg.get_value("epochs") / 20, 500),
                num_training_steps= len(self.dataloader_train) * self.cfg.get_value("epochs")
            )
               

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

            # Predict the noise residual w/o conditioning
            denoised_sample, noise = self.model(data_samples, None, condition)
            
            index = batch.pop("index", None)
            if index != None:
                loss = self.criterion(denoised_sample, index)
                loss += torch.nn.functional.mse_loss(noise, data_samples)
            else:
                if self.model.learn_snr:
                    l = torch.nn.MSELoss(reduction='none')
                    loss = torch.mean(l(denoised_sample, noise), dim=(-1, -2))
                    loss *= self.model.mean_snr
                    loss = loss.mean()
                else:
                    loss = self.criterion(denoised_sample, noise)

            total_loss += loss.item()

            loss.backward()
            if self.gradient_clipping > 0.:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update Exponential Moving Average of the model weights
            if self.ema:
                self.ema.step(self.model.parameters())

        if self.scheduler:
            self.scheduler.step()

        # Calculate average loss for the epoch
        average_loss = total_loss / len(self.dataloader_train)        
        
        return average_loss

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
            loss = torch.nn.functional.mse_loss(data_generated[:, :8*3], data_samples[:, :8*3])
            total_loss += loss.item()

        # Calculate average loss for the epoch
        average_loss = total_loss / len(self.dataloader_test)

        return average_loss
    
    def train(self):
        """
        Train the model for multiple epochs.
        """
        progress_bar = trange(self.epochs, desc = "Epochs")

        for epoch in progress_bar:
            train_loss = self._train_epoch()

            if epoch % self.ckpt_every and epoch > 0:
                self._save_model()

            self.train_loss_history.append(train_loss)
            self.logs_loss_dict["tr_loss"] = train_loss

            # Test
            test_loss = None
            if (self.dataloader_test != None and epoch > 0 and epoch % self.train_test_ratio == 0):
                test_loss = self._test_epoch()
                # Save model with best test loss
                if (epoch != 0 and test_loss < self.min_test_loss):
                    self._save_model()
                    self.min_test_loss = test_loss
    
             # Write logs and history
                self.logs_loss_dict["tst_loss"] = test_loss

            self.logger.write_scalars("Loss", self.logs_loss_dict, self.current_epoch)
            progress_bar.set_postfix(self.logs_loss_dict)

            self.current_epoch += 1

            if self.scheduler != None:
                self.scheduler.step()

        self._save_model()
        self.write_hparams(self.cfg.get_cfg_as_dict())
        self.write_results()
        self.save_training_curves()