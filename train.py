from typing import Callable, Optional, Tuple, Union
import torch
from litgpt import LLM
from litgpt.api import Preprocessor
from litgpt.data import Alpaca2k
import lightning as L
from litgpt.utils import chunked_cross_entropy

from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Union, Tuple


import torch
import lightning as L

from litgpt.config import name_to_config, Config
from litgpt.tokenizer import Tokenizer
from litgpt.prompts import (
    load_prompt_style,
    has_prompt_style,
    PromptStyle
)
from litgpt.utils import (
    auto_download_checkpoint,
    check_file_size_on_cpu_and_warn,
    chunked_cross_entropy,
    extend_checkpoint_dir,
    get_default_supported_precision,
    load_checkpoint,
)

from gpt import new_GPT
from utils import lightweight_entropy_estimator, parzen_entropy_estimator, pca_entropy_estimator, entropy_estimator_svd

class EntropyLLM(LLM):

    @classmethod
    def load(
        cls,
        model: str,
        init: Optional[Literal["pretrained", "random"]] = "pretrained",
        tokenizer_dir: Optional[Path] = None,
        access_token: Optional[str] = None,
        distribute: Optional[Literal["auto"]] = "auto"
    ) -> "LLM":
        """
        Loads the LLM from a local directory or model hub.

        Arguments
            model: A local path to a directory containing the model weights or a valid model name.
               You can get a list of valid model names via the `litgpt download list` command line argument.
            init: If "pretrained" (default), downloads the model from the HF Hub if a local model can't be found at the `model`
                directory name; otherwise loads the model from the local directory.
                If "random", initializes the `model` with random weights.
            tokenizer_dir: An optional tokenizer directory if `model` is not a checkpoint directory, or if a user
                wants to use a different tokenizer instead.
            access_token: Optional API token to access models with restrictions when using `init="pretrained"`.
            distribute: If "auto" (default), initializes the model on a single GPU if available and otherwise on the CPU.
                To have more control over the model distribution strategy and utilize multiple GPUs, you can set
                `llm = LLM.load(..., distribute=None)` and call `llm.distribute(...)` manually.
        """

        allowed_init = {"pretrained", "random"}

        if init == "pretrained":
            checkpoint_dir = auto_download_checkpoint(model_name=model, access_token=access_token, ignore_tokenizer_files=tokenizer_dir is not None)
            config = Config.from_file(checkpoint_dir / "model_config.yaml")

        elif init == "random":
            checkpoint_dir = None
            try:
                config = Config.from_name(model)
            except ValueError:
                print(f"Model name {model} is not supported.\n")
                available_models = "\n".join(sorted(name_to_config))
                print(f"Available values:\n{available_models}")
                return

        else:
            raise ValueError(f"Invalid init option: {init}. Must be one of {allowed_init}")

        torch.set_float32_matmul_precision("high")

        if tokenizer_dir is not None:
            tokenizer_dir = extend_checkpoint_dir(Path(tokenizer_dir))
            tokenizer = Tokenizer(tokenizer_dir)
        elif checkpoint_dir is not None:
            tokenizer = Tokenizer(checkpoint_dir)
        else:
            raise ValueError("Provide a path to a tokenizer directory via the `tokenizer_dir` setting.")

        if checkpoint_dir is not None:
            prompt_style = (
                load_prompt_style(checkpoint_dir)
                if has_prompt_style(checkpoint_dir)
                else PromptStyle.from_config(config)
            )
        else:
            prompt_style = PromptStyle.from_config(config)

        if distribute == "auto":
            if torch.cuda.is_available():
                accelerator = "cuda"
            elif torch.backends.mps.is_available():
                accelerator = "mps"
            else:
                accelerator = "cpu"

            fabric = L.Fabric(
                accelerator=accelerator,
                devices=1,
                precision=get_default_supported_precision(training=False),
            )

            with fabric.init_module(empty_init=False):
                model = new_GPT(config)
            model.eval()
            preprocessor = Preprocessor(tokenizer, device=fabric.device)

            if checkpoint_dir is not None:
                checkpoint_path = checkpoint_dir / "lit_model.pth"
                check_file_size_on_cpu_and_warn(checkpoint_path, fabric.device)
                load_checkpoint(fabric, model, checkpoint_path)

            model = fabric.setup_module(model)

        else:
            preprocessor = Preprocessor(tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")
            model = None
            fabric = None

        return cls(
            model=model, preprocessor=preprocessor, prompt_style=prompt_style,
            config=config, checkpoint_dir=checkpoint_dir, fabric=fabric, generate_strategy=None,
            kv_cache_initialized=False, fixed_kv_cache_size=False
        )
    
    def trainer_setup(self, trainer_ckpt: Optional[Path] = None) -> None:
        """Initializes the model checkpoint for PyTorch Lightning Trainer contexts"""
        self.model = new_GPT(self.config)

        if trainer_ckpt is not None:
            # strip the object name key from the state_dict
            state_dict = torch.load(trainer_ckpt, weights_only=True)["state_dict"]
            first_key = next(iter(state_dict))
            prefix = first_key.split(".")[0] + "."
            keys_to_modify = [key for key in state_dict if key.startswith(prefix)]
            for key in keys_to_modify:
                new_key = key.replace(prefix, "", 1)
                state_dict[new_key] = state_dict.pop(key)

            self.load_state_dict(state_dict, strict=True)

        elif self.checkpoint_dir is not None:
            state_dict = torch.load(self.checkpoint_dir / "lit_model.pth", weights_only=False)
            self.load_state_dict(state_dict, strict=False)

        else:
            raise ValueError(
                "No checkpoint found. Either provide a valid path via `trainer_ckpt` "
                "or ensure that `self.checkpoint_dir` points to a folder containing a `lit_model.pth` weight file."
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        logits, hs = self.model(input_ids, get_hidden_states=True)
        if target_ids is not None:
            if loss_fn is None:
                loss_fn = chunked_cross_entropy
            loss_ce = loss_fn(logits[..., :-1, :], target_ids[..., 1:])
            # mean of all hs
            entropy_loss = entropy_estimator_svd(hs[:, 0])
            total_loss = loss_ce + 0.1*entropy_loss
            return logits, total_loss, loss_ce, entropy_loss
        else:
            return logits



class LitEntropyLLM(L.LightningModule):
    def __init__(self, checkpoint_dir, tokenizer_dir=None, trainer_ckpt_path=None, using_entropy_regularizer=False):
        super().__init__()
            
        self.llm = EntropyLLM.load(checkpoint_dir, tokenizer_dir=tokenizer_dir, distribute=None)

        self.trainer_ckpt_path = trainer_ckpt_path

    def setup(self, stage):
        self.llm.trainer_setup(trainer_ckpt=self.trainer_ckpt_path)
        
    def training_step(self, batch):
        logits, total_loss, loss_ce, entropy_loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("train_loss_ce", loss_ce, prog_bar=True)
        self.log("train_entropy_loss", entropy_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch):
        logits, total_loss, loss_ce, entropy_loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("validation_loss_ce", loss_ce, prog_bar=True)
        self.log("validation_entropy_loss", entropy_loss, prog_bar=True)
        self.log("validation_total_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(self.llm.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]
    
class LitLLM(L.LightningModule):
    def __init__(self, checkpoint_dir, tokenizer_dir=None, trainer_ckpt_path=None, using_entropy_regularizer=False):
        super().__init__()
            
        self.llm = LLM.load(checkpoint_dir, tokenizer_dir=tokenizer_dir, distribute=None)

        self.trainer_ckpt_path = trainer_ckpt_path

    def setup(self, stage):
        self.llm.trainer_setup(trainer_ckpt=self.trainer_ckpt_path)
        
    def training_step(self, batch):
        logits, loss_ce = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("train_loss_ce", loss_ce, prog_bar=True)
        return loss_ce

    def validation_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("validation_loss_ce", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(self.llm.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]
    
def main():
    from lightning.pytorch.loggers import WandbLogger

    wandb_logger = WandbLogger(project="EntropyLLM", name="llm_entropy_sum")
    
    batch_size = 8
    accumulate_grad_batches = 1

    lit_model = LitEntropyLLM(checkpoint_dir="checkpoints/EleutherAI/pythia-70m")
    #lit_model = LitLLM(checkpoint_dir="checkpoints/EleutherAI/pythia-70m")
    data = Alpaca2k()

    data.connect(lit_model.llm.tokenizer, batch_size=batch_size, max_seq_length=512)

    trainer = L.Trainer(
        devices=1,
        accelerator="cuda",
        max_epochs=5,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
        logger=wandb_logger
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)

if __name__ == "__main__":
    main()