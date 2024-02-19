import os
import csv
import argparse
import torch.nn as nn
import time
import torch.nn.functional as F
import torch
import pandas as pd
import zipfile
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, AutoTokenizer
from transformers import BertPreTrainedModel, BertModel, EarlyStoppingCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import f1_score, accuracy_score
from SemEval.focal_loss import FocalLoss
from SemEval.focal_loss import BinaryDiceLoss
import SemEval.losses as losses

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
train = pd.read_csv("SemEval/dataset_semeval24_traindev/train.csv", encoding='utf-8')
dev = pd.read_csv("SemEval/dataset_semeval24_traindev/dev.csv", encoding='utf-8')
test = pd.read_csv("SemEval/dataset_semeval24_traindev/test.csv", encoding='utf-8')
train_dict = {'label': train['label'], 'explanation': train['explanation'], 'question': train['question'],
              'analysis': train['analysis'], 'completeanalysis': train['complete analysis'], 'answer': train['answer'],
              'idx': train['idx']}
dev_dict = {'label': dev['label'], 'explanation': dev['explanation'], 'question': dev['question'],
            'analysis': dev['analysis'], 'completeanalysis': dev['complete analysis'], 'answer': dev['answer'],
            'idx': dev['idx']}
test_dict = {'explanation': test['explanation'], 'question': test['question'], 'answer': test['answer'],
             'idx': test['idx']}


def sliding_window_ds_approach(df, window_size=50):
    dp = (
        []
    )  # table_of_content=["question", "answer", "label", "analysis", "complete analysis", "explanation", "idx"]
    for _, row in df.iterrows():
        (
            label,
            explanation,
            question,
            analysis,
            completeanalysis,
            answer,
            idx,
            idx_complete,
        ) = row
        cache = explanation + " | " + question
        cache = cache.split()  # Split on whitespace
        if len(cache) <= window_size:
            dp.append(
                (
                    explanation + " | " + question,
                    answer,
                    label,
                    analysis,
                    idx,
                    idx_complete,
                )
            )
        else:
            while len(cache) > window_size:
                sentence1 = " ".join(cache[: 150 + window_size])
                dp.append((sentence1, answer, label, analysis, idx, idx_complete))
                cache = cache[150:]

    return pd.DataFrame(
        dp, columns=["question", "answer", "label", "analysis", "idx", "idx_complete"]
    )


def sliding_window_ds_approach_keep_question(df, window_size=50):
    dp = (
        []
    )
    for index, row in df.iterrows():
        (
            label,
            explanation,
            question,
            analysis,
            completeanalysis,
            answer,
            idx,
            idx_complete,
        ) = row
        question = question + " | "
        question_len = len(question.split())
        cache = explanation
        cache = cache.split()  # Split on whitespace
        if len(cache) <= window_size:
            dp.append(
                (question + explanation, answer, label, analysis, idx, idx_complete)
            )
        else:
            while len(cache) > window_size:
                append_len = 400 - question_len
                append_len = append_len if append_len > 0 else 0
                if append_len <= 0:  # Debugging
                    print(
                        "ERROR: Append len is 0. Question: %s len question %s"
                        % (question, question_len)
                    )
                    exit()
                sentence1 = question + " ".join(cache[: append_len + window_size])
                dp.append((sentence1, answer, label, analysis, idx, idx_complete))
                cache = cache[append_len:]

    return pd.DataFrame(
        dp, columns=["question", "answer", "label", "analysis", "idx", "idx_complete"]
    )


def create_dataset(sliding_window="", train_dict=None, dev_dict=None, test_dict=None):
    train_data, dev_data, test_data = train_dict, dev_dict, test_dict
    train_data = pd.DataFrame(train_data)
    dev_data = pd.DataFrame(dev_data)
    test_data = pd.DataFrame(test_data)

    train_data["idx"] = [i for i in range(train_data["idx"].size)]
    dev_data["idx"] = [i for i in range(dev_data["idx"].size)]
    test_data["idx"] = [i for i in range(test_data["idx"].size)]

    train_data["idx_complete"] = [i for i in range(train_data["idx"].size)]
    dev_data["idx_complete"] = [i for i in range(dev_data["idx"].size)]
    test_data["idx_complete"] = [i for i in range(test_data["idx"].size)]

    train_data["label"] = [int(i) for i in train_data["label"]]
    dev_data["label"] = [int(i) for i in dev_data["label"]]

    if sliding_window == "simple":
        train_data = sliding_window_ds_approach(train_data)
        dev_data = sliding_window_ds_approach(dev_data)
        test_data = sliding_window_ds_approach(test_data)

        train_data["idx"] = [i for i in range(train_data["idx"].size)]  # Reset index
        dev_data["idx"] = [i for i in range(dev_data["idx"].size)]
        test_data["idx"] = [i for i in range(test_data["idx"].size)]

    elif sliding_window == "keep_question":
        train_data = sliding_window_ds_approach_keep_question(train_data)
        dev_data = sliding_window_ds_approach_keep_question(dev_data)
        test_data = sliding_window_ds_approach_keep_question(test_data)

        train_data["idx"] = [i for i in range(train_data["idx"].size)]  # Reset index
        dev_data["idx"] = [i for i in range(dev_data["idx"].size)]
        test_data["idx"] = [i for i in range(test_data["idx"].size)]

    dataset_train = Dataset.from_pandas(train_data)
    dataset_dev = Dataset.from_pandas(dev_data)
    dataset_test = Dataset.from_pandas(test_data)
    complete_ds = DatasetDict(
        {"train": dataset_train, "dev": dataset_dev, "test": dataset_test}
    )
    return complete_ds


def tokenize_function(example):
    return tokenizer(
        example["question"], example["answer"], truncation=True, max_length=512
    )


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print([(l, p) for l, p in list(zip(labels, preds)) if l == 1][:10])
    # calculate accuracy using sklearn's function
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    print("dev set len:", len(tokenized_datasets["dev"]), "pred len:", len(preds))
    print("F1:", f1, "Acc:", acc)
    return {
        "f1_score": f1,
    }


def combine_splitted_samples(dataset):
    ds_list = []
    temp_dict = {}
    for sample in dataset:
        if sample["idx_complete"] not in temp_dict:
            temp_dict[sample["idx_complete"]] = dict(
                map(lambda x: (x, []), sample.keys())
            )
        for key in temp_dict[sample["idx_complete"]].keys():
            temp_dict[sample["idx_complete"]][key].append(sample[key])

    for sample in temp_dict.values():
        ds_list.append(Dataset.from_dict(sample))
    return ds_list


def predict_on_dataset(list_dataset):
    preds = []
    gt = []
    for sub_dataset in list_dataset:
        sub_predicition = trainer.predict(sub_dataset)
        # calc the average of the predictions
        sub_predicition = sub_predicition.predictions.mean(axis=0)
        preds.append(sub_predicition.argmax())
        gt.append(sub_dataset[0]["label"])
    return preds, gt


def KL(input, target, reduction="sum"):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
                    F.softmax(target, dtype=torch.float32), reduction=reduction)
    return loss


class BertScratch(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.config = config
        self.alpha = 0.02
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        kl_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        kl_output = kl_outputs[1]
        kl_output = self.dropout(kl_output)
        kl_logits = self.classifier(kl_output)
        total_loss = None
        if labels is not None:
            loss_fct = BinaryDiceLoss(1, 2)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            fc_loss = loss_fct(kl_logits.view(-1, self.num_labels), labels.view(-1))
            kl_loss = (KL(logits, kl_logits, "sum") + KL(kl_logits, logits, "sum")) / 2.
            total_loss = loss + fc_loss + kl_loss
        return SequenceClassifierOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


if __name__ == '__main__':
    model_base_path = os.path.join(os.getcwd(), "data", "done", "model_output")
    parser = argparse.ArgumentParser(description="Evaluate LegalBert")
    parser.add_argument(
        "--dataset_type",
        type=str,
        help='Dataset type to use. choose from: "", "simple", or "keep_question"',
        default="",
    )
    parser.add_argument(
        "--model",
        type=str,
        help='BERT based model to use',
        default="roberta-base",
    )
    args = parser.parse_args()
    legal_ds = create_dataset(args.dataset_type, train_dict, dev_dict, test_dict)  # "", "simple" or "keep_question"
    checkpoint = args.model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    tokenized_datasets = legal_ds.map(tokenize_function, batched=True)
    samples = tokenized_datasets["train"]["input_ids"][:8]

    N = 3
    results = []
    for index in range(N):
        training_args = TrainingArguments(
            output_dir=os.path.join(model_base_path, checkpoint + "_finetune"),
            group_by_length=True,
            per_device_train_batch_size=4,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=100,
            fp16=True,
            learning_rate=3e-5,
            warmup_steps=10,
            gradient_accumulation_steps=2,
            logging_strategy="epoch",
            seed=index,
            load_best_model_at_end=True,
            greater_is_better=True,
            metric_for_best_model="f1_score",
        )
        model = BertScratch.from_pretrained(checkpoint)
        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["dev"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=-0.05)]
        )

        trainer.train()
        dev_dataset_list = combine_splitted_samples(tokenized_datasets["dev"])
        test_dataset_list = combine_splitted_samples(tokenized_datasets["test"])
        preds, gt = predict_on_dataset(dev_dataset_list)
        print("F1 Score (Macro)", f1_score(preds, gt, average="macro"))
        print("F1 Score (binary)", f1_score(preds, gt, average="micro"))
        print("Accuracy", accuracy_score(preds, gt))
        results.append(
            (
                f1_score(preds, gt, average="macro"),
                f1_score(preds, gt, average="micro"),
                accuracy_score(preds, gt),
            )
        )
    path = os.path.join(
        os.path.join(os.getcwd(), "data", "done"), checkpoint + "_finetune"
    )
    if args.dataset_type == "":
        run_name = "FT_QEA_DEV_TESTCONFS_" + str(int(time.time()))
    elif args.dataset_type == "simple":
        run_name = "FT_QEA_DEV_SWS_TESTCONFS_" + str(int(time.time()))
    elif args.dataset_type == "keep_question":
        run_name = "FT_QEA_DEV_SWA_TESTCONFS_" + str(int(time.time()))

    # Calc mean F1 macro and binary score
    f1_macro = np.mean([i[0] for i in results])
    f1_binary = np.mean([i[1] for i in results])
    accuracy = np.mean([i[2] for i in results])

    print(
        "Mean N=%s Score: F1 Score (Macro)" % N,
        f1_macro,
        "F1 Score (binary)",
        f1_binary,
        "Accuracy",
        accuracy,
    )

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, run_name), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["f1_macro", "f1_binary", "accuracy"])
        writer.writerows(results)

    evaluation_dataset = "SemEval/dataset_semeval24_traindev/dev.csv"
    df_eval_data = pd.read_csv(evaluation_dataset, index_col=0)


    def majority_baseline(eval_data):
        df_result = pd.DataFrame(index=eval_data.index)
        df_result['baseline'] = preds
        return df_result


    result_data = majority_baseline(df_eval_data)
    result_data.to_csv('majority_baseline.csv')
    zip = zipfile.ZipFile('example_submission_majority_baseline.zip', 'w', zipfile.ZIP_DEFLATED)
    zip.write('majority_baseline.csv')
    zip.close()
