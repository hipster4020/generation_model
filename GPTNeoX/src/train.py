import os
import hydra
import wandb

from transformers import (
    AutoTokenizer,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from dataloader import load

@hydra.main(config_name="config.yml")
def main(cfg):
    # wandb init
    # wandb.init(project=cfg.TRAINING.project_name)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.PATH.tokenizer)

    # dataloader
    train_dataset, eval_dataset = load(tokenizer, **cfg.DATASETS)
    print(f"train_dataset : {train_dataset}")

    # model
    print(f"vocab_size : {tokenizer.vocab_size}")
    print(f"bos_token_id : {tokenizer.bos_token_id}")
    print(f"eos_token_id : {tokenizer.eos_token_id}")

    model = GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **cfg.MODEL,
        )
    )
    print(f"model : {model}")

    # wandb
    if cfg.ETC.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = cfg.ETC.wandb_project

    # trainingargs setting
    args = TrainingArguments(
        do_train=True,
        do_eval=eval_dataset is not None,
        logging_dir=cfg.PATH.logging_dir,
        output_dir=cfg.PATH.checkpoint_dir,
        **cfg.TRAININGARGS,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=None,
        data_collator=default_data_collator,
    )

    # train
    trainer.train()

    # 
    trainer.save_model(cfg.PATH.output_dir)

    if cfg.ETC.get("wandb_project"):
        wandb.finish()

if __name__ == "__main__":
    main()