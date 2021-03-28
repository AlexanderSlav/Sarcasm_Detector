from transformers import AlbertForSequenceClassification, AlbertTokenizerFast
from dataset import SARCDataset
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd
import logging as logger
logger.basicConfig(level=logger.INFO)

RANDOM_SEED = 42


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def split_data(data, test_percent: float = 0.2):
    return train_test_split(
        data,
        test_size=test_percent,
        random_state=RANDOM_SEED
    )


def main():
    logger.info('Loading model...')
    model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')
    model.train().to('cuda')
    data = pd.read_csv("data/train-balanced-sarcasm.csv")
    truncated_data = data.sample(n=2500, random_state=1)
    data_train, data_test = split_data(truncated_data)
    tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
    train_dataset = SARCDataset(data_train, tokenizer)
    test_dataset = SARCDataset(data_test, tokenizer)
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=15,  # total # of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',
        dataloader_num_workers=8,
        logging_steps=50,
        save_steps=50,
        eval_steps=50,
        evaluation_strategy="epoch",

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()