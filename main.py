import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the model.")
    parser.add_argument(
        "--mode", choices=["train", "test"], required=True, help="Mode: train or test")
    args = parser.parse_args()

    if args.mode == "train":
        from train import train_model
        train_model()
    elif args.mode == "test":
        from test import test_model
        test_model()
