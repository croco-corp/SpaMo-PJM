import os

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback


def dump_test_outputs(save_dir, references, generated, logger=None, split: str = "test"):
    """Dump full reference/generated pairs to <save_dir>/<split>_outputs.txt and
    log a W&B table when a wandb logger is provided.

    Used by `on_test_epoch_end` (and optionally `on_validation_epoch_end`) — the
    paper's `LoggingCallback.log_generated_text` was deadcode (never wired into the
    Lightning module); this is the working replacement. File dump captures all pairs
    (1468 for PJM-MS test); W&B table is the same data for dashboard inspection.
    """
    pairs = list(zip(references, generated))

    try:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{split}_outputs.txt")
        with open(path, "w") as f:
            f.write(f"# {len(pairs)} pairs ({split})\n\n")
            for i, (ref, gen) in enumerate(pairs):
                f.write(f"[{i}]\nReference: {ref}\nGenerated: {gen}\n\n")
        print(f"Dumped {len(pairs)} pairs → {path}")
    except Exception as e:
        print(f"Could not dump {split}_outputs.txt: {e}")

    try:
        if logger is not None and hasattr(logger, "experiment"):
            import wandb
            table = wandb.Table(
                columns=["idx", "reference", "generated"],
                data=[[i, ref, gen] for i, (ref, gen) in enumerate(pairs)],
            )
            logger.experiment.log({f"{split}/translations": table})
            print(f"Logged {len(pairs)} translations to W&B {split}/translations")
    except Exception as e:
        print(f"Could not log W&B {split} table: {e}")


class LoggingCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__()

    def log_generated_text(
        self, 
        save_dir, 
        ids,
        vis_strings, 
        gloss_strings,
        generated_strings, 
        reference_strings, 
        prefix=None
    ):
        """
        Logs generated text for validation and testing.
        """
        save_dir = os.path.join(save_dir, "text")
        os.makedirs(save_dir, exist_ok=True)
        file_name = "outputs.txt"
        
        if prefix is not None:
            file_name = f"{prefix}-outputs.txt"
        
        if gloss_strings != []:
            with open(os.path.join(save_dir, file_name), "w") as file:
                for id, vis, gls, gen, ref in zip(ids, vis_strings, gloss_strings, generated_strings, reference_strings):
                    file.write(f"ID: {id}\nVis Token: {vis}\nGloss: {gls}\nReference: {ref}\nGenerated: {gen}\n\n")
        else:
            with open(os.path.join(save_dir, file_name), "w") as file:
                for id, vis, gen, ref in zip(ids, vis_strings, generated_strings, reference_strings):
                    file.write(f"ID: {id}\nVis Token: {vis}\nReference: {ref}\nGenerated: {gen}\n\n")

    def on_test_end(self, trainer, pl_module):
        ids = pl_module.id_list
        vis_strings = pl_module.vis_string_list
        glosses  = pl_module.gloss_list
        generated = pl_module.generated_text_list
        references = pl_module.reference_text_list

        self.log_generated_text(
            pl_module.logger.save_dir, ids, vis_strings, glosses, generated, references,  # type: ignore
        )


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0:
            if pl_module.global_step != 0:
                print("[INFO] Summoning checkpoint.")
                ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            # print("Project config")
            # print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            # print("Lightning config")
            # print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        