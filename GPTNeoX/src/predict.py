import hydra
from transformers import AutoTokenizer, AutoModelForCausalLM

@hydra.main(config_name="config.yml")
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.PATH.tokenizer)
    print(f"tokenizer : {tokenizer}")

    test_sentence = "안녕하세요." + tokenizer.eos_token + tokenizer.bos_token
    print(test_sentence)
    input_ids = tokenizer(test_sentence, return_tensors="pt").input_ids

    # model
    model = AutoModelForCausalLM.from_pretrained(cfg.PATH.output_dir)
    print(f"model : {model}")

    output = model.generate(input_ids)
    # huggingface model.generate top-k, beam search

    # predict
    print(output)
    print(f"output : {tokenizer.decode(output[0])}")

if __name__ == "__main__":
    main()