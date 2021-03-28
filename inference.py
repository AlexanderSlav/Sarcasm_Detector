from transformers import AlbertForSequenceClassification, AlbertTokenizerFast
import torch.nn.functional as f
import torch


def main():
    model = AlbertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=
                                                            "results/checkpoint-900")
    model.eval().to('cuda')
    tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
    text = "Ow really, that would work!"
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'].to('cuda')
    attention_mask = encoding['attention_mask'].to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        answers = f.softmax(outputs.logits, dim=1)
        print(answers)


if __name__ == "__main__":
    main()
