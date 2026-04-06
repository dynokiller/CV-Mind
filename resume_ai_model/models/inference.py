import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class DomainClassifier:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DomainClassifier, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        self.model_path = "models/domain_classifier"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            # return_all_scores ensures output format consistency
            self.pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, top_k=None)
            self.explainer = shap.Explainer(self.pipeline)
        except Exception as e:
            print(f"[Warning] Model load failed: {e}. Train the model first using training/train_model.py")
            self.model = None

    def predict(self, text: str):
        if not self.model:
            return "Model Not Loaded", 0.0, []
        
        # Determine the best prediction
        preds = self.pipeline(text[:2000])[0]
        best_pred = max(preds, key=lambda x: x['score'])
        
        # Explanations via SHAP
        shap_values = self.explainer([text[:500]]) # SHAP is computationally heavy, limiting string size
        
        feature_names = shap_values.data[0]
        values = shap_values.values[0]
        
        if len(values.shape) > 1:
            class_idx = [k for k, v in self.model.config.id2label.items() if v == best_pred['label']][0]
            values = values[:, class_idx]

        word_impacts = [{"word": str(w).strip(), "impact": round(float(v), 4)} for w, v in zip(feature_names, values) if str(w).strip()]
        # Filter valid interactions and sort by impact
        word_impacts = sorted([w for w in word_impacts if w["impact"] > 0], key=lambda x: x['impact'], reverse=True)[:10]

        return best_pred['label'], best_pred['score'], word_impacts

# Singleton loading at startup
classifier = DomainClassifier()
