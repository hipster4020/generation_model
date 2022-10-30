import hydra
import wandb

from transformers import AutoTokenizer
from dataloader import get_dataloader, load

@hydra.main(config_name="config.yml")
def main(cfg):
    # wandb init
    # wandb.init(project=cfg.TRAINING.project_name)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.model_name)

    # dataloader
    train_dataset, eval_dataset = load(**cfg.DATASETS)    
    train_dataloader = get_dataloader(train_dataset, **cfg.DATALOADER)
    print(f"train_dataloader : {train_dataloader}")
    eval_dataloader = get_dataloader(eval_dataset, **cfg.DATALOADER)


if __name__ == "__main__":
    main()