import numpy as np
import torch
import torch.nn.functional as F

class MedicalHallucinationDetector:

    def __init__(self, med_tokenizer, med_model, nli_tokenizer, nli_model):
        self.med_tokenizer = med_tokenizer
        self.med_model = med_model
        self.nli_tokenizer = nli_tokenizer
        self.nli_model = nli_model

    # =========================
    # EMBEDDING (FIXED)
    # =========================
    def embed(self, text):
        inputs = self.med_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.med_model.device)

        with torch.no_grad():
            outputs = self.med_model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        return summed / summed_mask

    # =========================
    # NLI
    # =========================
    def nli_scores(self, ai, final):
        inputs = self.nli_tokenizer(
            ai,
            final,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.nli_model.device)

        with torch.no_grad():
            logits = self.nli_model(**inputs).logits

        probs = torch.softmax(logits, dim=1)

        contradiction = probs[0][0].item()
        entailment = probs[0][2].item()

        return entailment, contradiction

    # =========================
    # PREDICT
    # =========================
    def predict(self, ai_text, final_text):

        # Normalize embeddings
        emb1 = F.normalize(self.embed(ai_text), p=2, dim=1)
        emb2 = F.normalize(self.embed(final_text), p=2, dim=1)

        # Cosine similarity (stable)
        sim = torch.sum(emb1 * emb2, dim=1).item()

        # Length-aware scaling
        len_ratio = min(len(ai_text), len(final_text)) / max(len(ai_text), len(final_text))
        sim = sim * (0.8 + 0.2 * len_ratio)

        # NLI
        entail, contra = self.nli_scores(ai_text, final_text)

        # Consistency
        consistency = 0.55 * sim + 0.25 * entail - 0.2 * contra
        consistency = float(np.clip(consistency, 0, 1))

        # Risk
        risk = 0.4 * (1 - sim) + 0.3 * contra + 0.3 * (1 - entail)
        risk = float(np.clip(risk, 0, 1))

        return {
            "consistency_score": consistency,
            "risk_score": risk,
            "similarity": sim,
            "entailment": entail,
            "contradiction": contra
        }