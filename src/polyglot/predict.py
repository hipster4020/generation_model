import hydra
from transformers import AutoTokenizer, AutoModelForCausalLM

@hydra.main(config_name='config.yml')
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.model_name)

    test_sentence = "안녕하세요. 저는 기타를 잘 칩니다."
    input_ids = tokenizer(test_sentence, return_tensors="pt").input_ids

    # model
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL.model_name)
    output = model.generate(input_ids)
    
    # predict
    print(output)
    print(f"output : {tokenizer.decode(output[0])}")

if __name__ == "__main__":
    main()