import datasets


from transformers import Trainer, TrainingArguments,EvalPrediction, BigBirdTokenizer, BigBirdForSequenceClassification
from datasets import load_metric
import numpy as np
import pandas as pd


train_data = datasets.load_dataset('csv', data_files='train.csv')['train']
val_data = datasets.load_dataset('csv', data_files='val.csv')['train']
test_data = datasets.load_dataset('csv', data_files='test.csv')['train']





# Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained("usc-isi/sbert-roberta-large-anli-mnli-snli")
# model = AutoModel.from_pretrained("usc-isi/sbert-roberta-large-anli-mnli-snli")
# model.save_pretrained("./models/sbert/")
# tokenizer.save_pretrained("./models/sbert/")
tokenizer = BigBirdTokenizer.from_pretrained(r"E:\code\FCC_Transformer\models\tokenizer")
model = BigBirdForSequenceClassification.from_pretrained(r"E:\code\FCC_Transformer\models\model")

def tokenization(batched_text):
    return tokenizer(batched_text['text_data'], padding = 'max_length', truncation=True, max_length = 1024)

train_data = train_data.map(tokenization, batched = True, batch_size = len(train_data))
val_data = val_data.map(tokenization, batched = True, batch_size = len(val_data))
test_data = test_data.map(tokenization, batched = True, batch_size = len(test_data))

# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)
#
# # Perform pooling. In this case, max pooling.
# sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
#
# print("Sentence embeddings:")
# print(sentence_embeddings)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, log_loss
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

training_args = TrainingArguments(
    output_dir = './media/sbert',
    num_train_epochs = 50,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 32,
    per_device_eval_batch_size= 8,
    # evaluation_strategy = "epoch",
    # save_strategy= "epoch",
    evaluation_strategy = "epoch",
    save_strategy="epoch",
    # disable_tqdm = False,
    load_best_model_at_end=True,
    # warmup_steps=160,
    weight_decay=0.01,
    # logging_steps = 4,
    learning_rate = 1e-4,
    fp16 = True,
    logging_dir='./media/logs2',
    # dataloader_num_workers = 0,
    run_name = 'sbert_classification_1e5',
    save_total_limit = 3,
)

from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 2 labels with different weights)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data
)

trainer.train()
trainer.save_model("./BigBirdData/Model")
pred = trainer.predict(test_data)
np.save("pred_bigbird.npy", pred.predictions)