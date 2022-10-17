import hydra
from transformers import AutoTokenizer, AutoModelForCausalLM

@hydra.main(config_name='config.yml')
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.model_name)
    print(f"tokenizer : {tokenizer}")

    test_sentence = "안녕하세요. 테스트입니다."
    print(f"encode : {tokenizer.encode(test_sentence)}")
    print(f"tokenize : {tokenizer.tokenize(test_sentence)}")

    print(f"decode : {tokenizer.decode(tokenizer.encode(test_sentence))}")
    
    # model
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL.model_name)
    print(f"model : {model}")



if __name__ == "__main__":
    main()