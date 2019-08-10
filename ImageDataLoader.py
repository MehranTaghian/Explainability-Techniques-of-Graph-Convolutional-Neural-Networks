
import os

from torchvision.datasets.folder import ImageFolder, default_loader


################################################################################
# PyTorch
class ImageRegionData(ImageFolder):
    """

    """

    def __init__(
            self,
            root: str,
            # image_id_dir: int,
            transform=None,
            target_transform=None,
            loader=default_loader
    ):
        self.root = os.path.expanduser(root)

        path = os.path.join(self.root)
        print(f"Loading data from {path}.")
        assert os.path.isdir(path), "is not valid."

        super().__init__(path, transform, target_transform, loader)
