import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from slr.app.realtime_app import RealtimeApp
from slr.config import Config
from slr.data.collector import DatasetCollector
from slr.data.dataset import SignDataset
from slr.data.importers.landmark_table import LandmarkTableImporter
from slr.data.repository import DatasetRepository
from slr.inference.predictor import SignPredictor
from slr.inference.sentence import SentenceBuilder
from slr.keypoints.extractor import KeypointExtractor
from slr.models.arch import build_model
from slr.training.trainer import Trainer
from slr.training.seed import set_seed


def cmd_train(args, cfg: Config) -> None:
    set_seed(args.seed)
    repo = DatasetRepository(cfg.data_dir)
    print(f"Signs: {len(repo.label_names)} | signers: {repo.signers} | samples: {len(repo.records)}")

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


def cmd_collect(args, cfg: Config) -> None:
    extractor = KeypointExtractor()
    try:
        DatasetCollector(cfg, extractor).collect(args.label, args.signer, args.n)
    finally:
        extractor.close()


def cmd_import_table(args, cfg: Config) -> None:
    importer = LandmarkTableImporter(cfg, Path(args.path))
    saved = importer.run()
    print(f"Saved {saved} new samples to {cfg.data_dir}")


def add_dataset_arg(p: argparse.ArgumentParser, required: bool = False) -> None:
    """--dataset picks the data folder and the checkpoint folder (see Config.for_dataset)."""
    p.add_argument("--dataset", default="raw", required=required,
                   help="dataset name: data in data/<name>, checkpoints in checkpoints/<name>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign language recognition")
    sub = parser.add_subparsers(dest="command", required=True)
    models = ["lstm", "gru", "tcn"]

    p = sub.add_parser("train", help="train a model")
    p.add_argument("--model", default="lstm", choices=models)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-signers", default="", help="comma-separated held-out signer ids")
    p.add_argument("--seed", type=int, default=0, help="random seed for reproducible runs")
    add_dataset_arg(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("run", help="run live recognition from the webcam")
    p.add_argument("--model", default="lstm", choices=models)
    add_dataset_arg(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("collect", help="record examples of a sign from the webcam")
    p.add_argument("--label", required=True)
    p.add_argument("--signer", required=True)
    p.add_argument("--n", type=int, default=30)
    add_dataset_arg(p)
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("import-table", help="import a long-format landmark CSV")
    p.add_argument("--path", required=True, help="path to the CSV file")
    add_dataset_arg(p, required=True)
    p.set_defaults(func=cmd_import_table)

    args = parser.parse_args()
    args.func(args, Config().for_dataset(args.dataset))


if __name__ == "__main__":
    main()