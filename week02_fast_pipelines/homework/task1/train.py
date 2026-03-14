import torch
from torch import nn
from tqdm.auto import tqdm

from unet import Unet

from dataset import get_train_data


class StaticScaler:
    def __init__(self, scale_factor: float = 2.0 ** 10):
        self.scale_factor = scale_factor

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale_factor

    def step(self, optimizer: torch.optim.Optimizer):
        inv_scale = 1.0 / self.scale_factor
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.mul_(inv_scale)
        optimizer.step()

    def update(self):
        pass


class DynamicScaler:

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.scale_factor = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._steps_since_last_overflow = 0
        self._found_inf = False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale_factor

    def step(self, optimizer: torch.optim.Optimizer):
        inv_scale = 1.0 / self.scale_factor
        self._found_inf = False
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        if not torch.isfinite(param.grad).all():
                            self._found_inf = True
                            optimizer.zero_grad()
                            return
                        param.grad.mul_(inv_scale)
        optimizer.step()

    def update(self):
        if self._found_inf:
            self.scale_factor *= self.backoff_factor
            self._steps_since_last_overflow = 0
        else:
            self._steps_since_last_overflow += 1
            if self._steps_since_last_overflow >= self.growth_interval:
                self.scale_factor *= self.growth_factor
                self._steps_since_last_overflow = 0
    


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
) -> None:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device.type, dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(scaler=None):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_epoch(train_loader, model, criterion, optimizer, device=device, scaler=scaler)


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "static"

    if mode == "static":
        print("Training with StaticScaler")
        train(scaler=StaticScaler())
    elif mode == "dynamic":
        print("Training with DynamicScaler")
        train(scaler=DynamicScaler())
    elif mode == "none":
        print("Training without scaler (AMP only)")
        train(scaler=None)
    else:
        print(f"Unknown mode: {mode}. Use 'static', 'dynamic', or 'none'.")
