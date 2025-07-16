import argparse
import yaml

from model.model import SimCLR
from preprocessing import SimCLRDataModule
from train import Trainer

def main():
    parser = argparse.ArgumentParser(description="SimCLR Implementation CLI")
    parser.add_argument("--config", type=str, default="config/config.yml", help="Path to config file")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--test", action="store_true", help="Run test loop (WIP)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Setup components
    datamodule = SimCLRDataModule(config=config)
    trainer = Trainer(config)

    if args.train:
        print("Starting training...")
        datamodule.setup(stage="fit")
        trainer.train(
            train_loader=datamodule.train_dataloader(),
            val_loader=datamodule.val_dataloader()
        )

    elif args.eval:
        print("Running evaluation...")
        datamodule.setup(stage="validate")
        trainer.validate(datamodule.val_dataloader())

    elif args.test:
        print("Test mode selected.")
        datamodule.setup(stage="test")
        trainer.test()

    else:
        print("No mode selected. Use --train or --eval or --test.")


if __name__ == "__main__":
    main()
