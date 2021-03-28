from transformers import AlbertForSequenceClassification, AlbertTokenizerFast
import torch.nn.functional as f
import torch
import argparse
import logging as logger
logger.basicConfig(level=logger.INFO)
answer_map = \
    {
        0: "Non-sarcastic",
        1: "Sarcastic"
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, help="Input text for inference")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info('Loading model!')
    model = AlbertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=
                                                            "results/latest")
    model.eval().to('cuda')
    tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
    text = args.t
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'].to('cuda')
    attention_mask = encoding['attention_mask'].to('cuda')
    logger.info('Inference model!')
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        answers = f.softmax(outputs.logits, dim=1)
        logger.info(answer_map[torch.argmax(answers).item()])


if __name__ == "__main__":
    main()
