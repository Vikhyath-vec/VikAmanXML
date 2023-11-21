from transformers import GPT2LMHeadModel
import torch.nn as nn
import torch
import copy

class UIPrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, freezeLM=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt()
        return model

    def init_prompt(self):
        self.src_len = 2
        emsize = self.transformer.wte.weight.size(1) # 768
        kgat_emsize = emsize
        self.user_embeddings = nn.Linear(kgat_emsize, emsize)
        self.item_embeddings = nn.Linear(kgat_emsize, emsize)
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, user, item, text, mask, ignore_index=-100):
        u_src = self.user_embeddings(user)
        i_src = self.item_embeddings(item)
        w_src = self.transformer.wte(text)
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), w_src], 1)
        
        if mask is None:
            return super().forward(inputs_embeds=src)
        else:
            device = user.device
            batch_size = user.size(0)
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))
            prediction = torch.cat([pred_left, pred_right], 1)
            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)
        
class ContinuousPromptLearning(UIPrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)