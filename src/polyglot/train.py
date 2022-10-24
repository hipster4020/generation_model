import hydra
from transformers import AutoTokenizer, AutoModelForCausalLM

@hydra.main(config_name='config.yml')
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.model_name)

    test_sentence = "안녕하세요. 테스트입니다."
    input_ids = tokenizer(test_sentence, return_tensors="pt").input_ids


    print(f"input_ids : {input_ids}")

    # model
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL.model_name)
    print(model(input_ids))




if __name__ == "__main__":
    main()