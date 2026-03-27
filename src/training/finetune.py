from src.model.processor import ReVisionProcessor
from src.model.revision_model import ReVisionForConditionalGeneration
from src.data.dataset import RevisionRewriteDataset
from src.training.args import get_args_fine_tuning
from transformers import (
    TrainingArguments,
    Trainer,
)
import numpy as np
import torch
import os
import random
from PIL import Image

# MODEL_ID = "anonymoususerrevision/ReVision-171M-224px-random"
MODEL_ID = "anonymoususerrevision/ReVision-250M-256-16"


def set_seed(seed):
    """Set seed for reproducibility"""
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disable to ensure reproducibility


def main(args):
    set_seed(42)
    use_auth_token = os.getenv("HF_TOKEN")

    model = ReVisionForConditionalGeneration.from_pretrained(
        MODEL_ID, use_auth_token=use_auth_token
    )

    processor = ReVisionProcessor.from_pretrained(
        MODEL_ID, use_auth_token=use_auth_token
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    dataset1 = RevisionRewriteDataset(
        split="train", use_auth_token=use_auth_token, processor=processor
    )

    for param in model.vision_tower.parameters():
        param.requires_grad = False

    trainer_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        remove_unused_columns=args.remove_unused_columns,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta2=args.adam_beta2,
        evaluation_strategy="no",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        optim=args.optim,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        output_dir=args.output_dir,
        bf16=False,
        fp16=True,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        max_grad_norm=1.0,
        max_steps=-1
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset1,
        data_collator=dataset1.collate_fn,
        args=trainer_args,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    # model.push_to_hub(
    #     "anonymoususerrevision/ReVision-250M-256-16-baseline", use_auth_token=use_auth_token
    # )
    # processor.push_to_hub(
    #     "anonymoususerrevision/ReVision-250M-256-16-baseline", use_auth_token=use_auth_token
    # )

    # print("training complete")


if __name__ == "__main__":

    args = get_args_fine_tuning()
    main(args)
