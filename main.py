from torch.utils.data import DataLoader

import argparse

from slr.config import Config
from slr.data.dataset import SignDataset
from slr.data.repository import DatasetRepository
from slr.models.arch import build_model
from slr.training.trainer import Trainer
from slr.app.realtime_app import RealtimeApp
from slr.inference.predictor import SignPredictor
from slr.inference.sentence import SentenceBuilder
from slr.keypoints.extractor import KeypointExtractor


def cmd_train(args, cfg: Config) -> None:
    repo = DatasetRepository(cfg.data_dir)
    print(f"Signs: {len(repo.label_names)} | signers: {len(repo.signers)} | samples: {len(repo.records)}")

    # person-independent split if --val-signers is given, otherwise random
    if args.val_signers:
        train_records, val_records = repo.split_by_signer(args.val_signers.split(","))
    else:
        train_records, val_records = repo.split_random()
    print(f"Train: {len(train_records)} | val: {len(val_records)}")

    # augmentation and shuffling for train only
    train_loader = DataLoader(SignDataset(train_records, cfg.sequence_length, augment=True),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SignDataset(val_records, cfg.sequence_length),
                            batch_size=args.batch_size)

    model = build_model(args.model, cfg.num_features, len(repo.label_names))
    trainer = Trainer(model, args.model)
    trainer.fit(train_loader, val_loader, args.epochs)
    trainer.save_checkpoint(cfg.checkpoint_for(args.model), repo.label_names,
                            cfg.sequence_length, cfg.num_features)
    print(f"Best val accuracy: {trainer.best_val_acc:.3f} -> {cfg.checkpoint_for(args.model)}")

def cmd_run(args, cfg: Config) -> None:
    extractor = KeypointExtractor()
    try:
        predictor = SignPredictor(cfg.checkpoint_for(args.model),
                                  cfg.confidence_threshold, cfg.smoothing_window)
        sentence = SentenceBuilder(cfg.word_cooldown_sec)
        RealtimeApp(cfg, extractor, predictor, sentence).run()
    finally:
        extractor.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Sign language recognition")
    sub = parser.add_subparsers(dest="command", required=True)
    models = ["lstm", "gru", "tcn"]

    p = sub.add_parser("train", help="train a model")
    p.add_argument("--model", default="lstm", choices=models)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-signers", default="", help="comma-separated held-out signer ids")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("run", help="run live recognition from the webcam")
    p.add_argument("--model", default="lstm", choices=models)
    p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args, Config())


if __name__ == "__main__":
    main()