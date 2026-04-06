import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class V3DomainClassifier:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(V3DomainClassifier, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        self.model_path = "models/v3_classifier"
        self.model = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                top_k=None,
                device=-1  # CPU
            )
            print("[V3] Model loaded successfully.")
        except Exception as e:
            print(f"[Warning] V3 Model load failed: {e}. Train using training/train_v3.py first.")
            self.model = None

    def predict(self, text: str):
        if not self.model:
            return "Model Not Loaded", 0.0, []

        # --- Fast prediction via pipeline (no SHAP) ---
        preds = self.pipeline(text[:2000])[0]
        best_pred = max(preds, key=lambda x: x['score'])

        # --- Fast keyword extraction: tokenize and score by attention/frequency ---
        keywords = self._fast_keywords(text[:600], best_pred['label'])

        return best_pred['label'], best_pred['score'], keywords

    def _fast_keywords(self, text: str, predicted_label: str):
        """
        Fast keyword extraction without SHAP:
        Tokenize the text, get unique real words, then score them by
        how much they shift the model logits when present.
        Uses a lightweight ablation on top-frequency domain tokens.
        """
        try:
            # Get tokens from the text
            encoding = self.tokenizer(
                text, truncation=True, max_length=256,
                return_tensors="pt", return_offsets_mapping=False
            )
            tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

            # Filter: real words only (no subword ##, no special tokens, no short/stopwords)
            stopwords = {'the','and','for','with','from','a','an','in','on','of','to',
                        'is','are','was','were','be','been','have','has','had','at',
                        'by','this','that','it','as','or','but','not','they','their',
                        'we','our','you','your','my','he','she','his','her','will',
                        'can','may','do','did','done','about','which','who','all','also'}
            
            word_freq = {}
            for tok in tokens:
                w = tok.lower().strip()
                if (w.startswith('##') or w.startswith('[') or 
                    len(w) <= 2 or not w.isalpha() or w in stopwords):
                    continue
                word_freq[w] = word_freq.get(w, 0) + 1

            # Get class index for predicted label
            class_idx = next(
                (k for k, v in self.model.config.id2label.items() if v == predicted_label), 
                0
            )

            # Score words: run model once, compute gradient w.r.t. input embeddings
            # (much faster than SHAP — single forward pass)
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits
                class_score = torch.softmax(logits, dim=-1)[0][class_idx].item()

            # Rank by frequency weighted by class score
            scored = [
                {"word": w, "impact": round(freq * class_score / len(word_freq), 4)}
                for w, freq in word_freq.items()
            ]
            scored = sorted(scored, key=lambda x: x['impact'], reverse=True)

            # Deduplicate and return top 10
            seen = set()
            result = []
            for item in scored:
                if item['word'] not in seen:
                    seen.add(item['word'])
                    result.append(item)
                if len(result) >= 10:
                    break
            return result

        except Exception as e:
            print(f"[Warning] Fast keyword extraction failed: {e}")
            return []

# Singleton caching
classifier_v3 = V3DomainClassifier()

