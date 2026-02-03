import os
import nltk
from pathlib import Path
from collections import defaultdict
from config.settings import Config

# Download tokenizer once
nltk.download("punkt", quiet=True)


class DataLoader:
    """
    Loads and processes medical text data from /data directory.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or Config.DATA_DIR)

    def load_documents(self):
        """
        Load documents from /data and split into sentences.

        Returns:
            texts: list[str] - list of sentences
            diseases: list[str] - disease key per sentence
            metadata: list[dict] - additional metadata per sentence
        """
        texts, diseases, metadata = [], [], []

        for disease_key in Config.IMAGE_CLASSES:
            file_path = self.data_dir / f"{disease_key}.txt"
            if not file_path.exists():
                print(f"[WARNING] Missing data file: {file_path}")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                continue

            sentences = nltk.sent_tokenize(content)
            for sentence in sentences:
                texts.append(sentence)
                diseases.append(disease_key)
                metadata.append({"disease": disease_key})

        return texts, diseases, metadata

    def get_statistics(self):
        texts, diseases, _ = self.load_documents()
        dist = defaultdict(int)

        for d in diseases:
            dist[d] += 1

        return {
            "total_sentences": len(texts),
            "total_diseases": len(dist),
            "avg_sentence_length": sum(len(t) for t in texts) / max(len(texts), 1),
            "disease_distribution": dict(dist),
        }
