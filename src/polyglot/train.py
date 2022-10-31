import hydra
import wandb

from transformers import AutoTokenizer
from dataloader import load

@hydra.main(config_name="config.yml")
def main(cfg):
    # wandb init
    # wandb.init(project=cfg.TRAINING.project_name)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.model_name)

    # dataloader
    train_dataset, eval_dataset = load(tokenizer, **cfg.DATASETS)

if __name__ == "__main__":
    main()