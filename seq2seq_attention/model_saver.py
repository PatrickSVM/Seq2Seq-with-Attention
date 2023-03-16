import torch
import os


class SaveBestModel:
    """
    Class to save the best model while training based on val
    loss
    """

    def __init__(self, out_dir):
        self.best_valid_loss = 100000
        self.full_out_dir = f"experiments/{out_dir}/best_model.pt"

        # Create folder
        if not os.path.exists(f"experiments/{out_dir}"):
            os.makedirs(f"experiments/{out_dir}")

    def __call__(
        self,
        val_loss,
        epoch,
        model,
    ):
        if val_loss < self.best_valid_loss:
            self.best_valid_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                },
                self.full_out_dir,
            )
