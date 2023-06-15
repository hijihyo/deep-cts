def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    # environment-specific
    parser.add_argument("--data-path", type=str, default="data/")
    parser.add_argument("--logs-path", type=str, default="logs/")
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-mps-device", action="store_true", default=False)
    # model-specific
    parser.add_argument("--model", type=str, default="model")
    parser.add_argument("--variant", type=str, default="variant")
    parser.add_argument("--weights", type=str, default="none")
    parser.add_argument("--num-classes", type=int, default=4)
    # data-specific
    parser.add_argument("--dataset", type=str, default="ctssev")
    parser.add_argument("--num-splits", type=int, default=5)  # fixed
    parser.add_argument("--num-repeats", type=int, default=10)  # fixed
    parser.add_argument("--nth-fold", type=int, default=19)  # fixed
    parser.add_argument("--per-device-train-batch-size", type=int, default=32)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    # training-specific
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--num-train-epochs", type=float, default=500.0)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")

    return parser.parse_args()


CONFIG_MAPPING = {
    # Only Thenar
    "onlyt_convnext": "OnlyTConvNeXtConfig",
    "onlyt_efficientnet": "OnlyTEfficientNetConfig",
    "onlyt_mobilenet": "OnlyTMobileNetConfig",
    "onlyt_resnet": "OnlyTResNetConfig",
    "onlyt_swint": "OnlyTSwinTConfig",
    "onlyt_vit": "OnlyTViTConfig",
    # Thenar and Hypothenar
    "tht_convnext_concat": "THTConvNeXtConcatConfig",
    "tht_efficientnet_concat": "THTEfficientNetConcatConfig",
    "tht_mobilenet_concat": "THTMobileNetConcatConfig",
    "tht_resnet_concat": "THTResNetConcatConfig",
    "tht_swint_concat": "THTSwinTConcatConfig",
    "tht_vit_concat": "THTViTConcatConfig",
}
MODEL_MAPPING = {
    # Only Thenar
    "onlyt_convnext": "OnlyTConvNeXt",
    "onlyt_efficientnet": "OnlyTEfficientNet",
    "onlyt_mobilenet": "OnlyTMobileNet",
    "onlyt_resnet": "OnlyTResNet",
    "onlyt_swint": "OnlyTSwinT",
    "onlyt_vit": "OnlyTViT",
    # Thenar and Hypothenar
    "tht_convnext_concat": "THTConvNeXtConcat",
    "tht_efficientnet_concat": "THTEfficientNetConcat",
    "tht_mobilenet_concat": "THTMobileNetConcat",
    "tht_resnet_concat": "THTResNetConcat",
    "tht_swint_concat": "THTSwinTConcat",
    "tht_vit_concat": "THTViTConcat",
}
DATASET_MAPPING = {"ctsdiag": "get_kfold_ctsdiag", "ctssev": "get_kfold_ctssev"}


if __name__ == "__main__":
    import logging
    import os
    import sys
    from datetime import datetime
    from shutil import copytree, ignore_patterns

    import evaluate
    import numpy as np
    import pandas as pd
    from pytz import timezone
    from torchvision import transforms as T
    from transformers import Trainer, TrainingArguments

    import data
    import models
    import utils

    args = parse_args()

    model_config = getattr(models, CONFIG_MAPPING[args.model])(
        args.variant,
        args.weights,
        args.num_classes,
    )
    model = getattr(models, MODEL_MAPPING[args.model])(model_config)

    assert os.path.exists(args.logs_path)
    current_datetime = datetime.now(timezone("Asia/Seoul"))
    run_path = os.path.join(
        args.logs_path, args.name, current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    )
    os.makedirs(run_path)
    copytree("src/", os.path.join(run_path, "src"), ignore=ignore_patterns("__pycache__"))

    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")

    fileHandler = logging.FileHandler(os.path.join(run_path, "dl_cv_main.log"))
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    stdHandler = logging.StreamHandler()
    stdHandler.setFormatter(formatter)
    logger.addHandler(stdHandler)

    logger.info(
        f"Command: CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')} python {' '.join(sys.argv)}"  # noqa: E501
    )
    logger.info(f"Path: {run_path}")
    logger.info("Arguments:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    training_args = TrainingArguments(
        output_dir=run_path,
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=100,
        log_level="info",
        logging_dir=run_path,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        seed=args.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        use_mps_device=args.use_mps_device,
    )

    # mean = (0.1651, 0.1651, 0.1651)
    # std = (0.1553, 0.1553, 0.1553)
    mean = (0.5, 0.5, 0.5)
    std = (0.25, 0.25, 0.25)
    train_transform = T.Compose(
        [
            utils.ToRGB(),
            T.RandomResizedCrop(224, antialias=True),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )  # RandAugment, TrivialAugmentWide, AugMix, AutoAugmentPolicy, RandomErasing
    eval_transform = T.Compose(
        [
            utils.ToRGB(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    train_dataset, test_dataset = getattr(data.functional, DATASET_MAPPING[args.dataset])(
        args.data_path,
        args.num_splits,
        args.num_repeats,
        args.nth_fold,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    score_names = ["accuracy"]
    metrics = evaluate.combine(score_names)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metrics.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_scores = trainer.evaluate(test_dataset, metric_key_prefix="test")
    logger.info("Test scores:")
    for score_name in ["loss", *score_names]:
        key = f"test_{score_name}"
        logger.info(
            f"  {score_name}: {np.mean(test_scores[key]):.4f} ({np.std(test_scores[key]):.4f})"
        )

    results = {
        "run_path": run_path,
        "repeat": 1,
    }
    for score_name in ["loss", *score_names]:
        key = f"test_{score_name}"
        results[key] = [test_scores[key]]

    df = pd.DataFrame.from_dict(results)
    df.to_excel(
        os.path.join(run_path, "results.xlsx"),
        sheet_name="Execution Results",
        index=False,
        columns=["run_path", "repeat"] + [f"test_{s}" for s in ["loss", *score_names]],
    )
