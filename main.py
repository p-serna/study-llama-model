from model.model import CustomGPT2Model, config, tokenizer
import pytorch_lightning as pl
from transformers import AdamW
from data.dataset import lm_datasets

class LanguageModel(pl.LightningModule):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        # Shift the input_ids one position to the right for labels.
        labels = batch["input_ids"].clone().detach()
        labels[:, :-1] = batch["input_ids"][:, 1:]
        
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=labels)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["input_ids"].clone().detach()
        labels[:, :-1] = batch["input_ids"][:, 1:]
        
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=labels)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)

model = CustomGPT2Model(config)
language_model = LanguageModel(model, tokenizer)

trainer = pl.Trainer(gpus=1, max_epochs=5)
trainer.fit(language_model, lm_datasets['train'], lm_datasets['valid'])
