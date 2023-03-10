import logging
from pathlib import Path
import click

import torch
import torchvision

# let's output the info
logging.basicConfig(level=logging.INFO)


@click.command()
@click.argument("input_dir", type=click.Path(exists=False))
@click.argument("output_dir", type=click.Path(exists=False))
def encode(input_dir, output_dir):
    print(input_dir)
    print(output_dir)

    model = torchvision.models.resnet50(pretrained=False)
    model.load_state_dict(torch.load("/input/checkpoint_100.tar"))
    model.eval()


if __name__ == "__main__":
    encode()
