"""FastText déterministe : workers=1 + seed fixe (pour reproductibilité)."""
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec


class EpochSaver(CallbackAny2Vec):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    def on_epoch_end(self, model):
        model.save(f'fasttext_{self.dataset_name}.model')


class EpochLogger(CallbackAny2Vec):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
    def on_epoch_begin(self, model):
        print(f"{self.dataset_name} - FastText Epoch start")
    def on_epoch_end(self, model):
        print(f"{self.dataset_name} - FastText Epoch end")


def train_fasttext(phrases, dataset_name,
                   vector_size=30, window=5, min_count=1,
                   workers=1, epochs=300, seed=42,
                   min_n=3, max_n=6):
    saver = EpochSaver(dataset_name)
    logger = EpochLogger(dataset_name)
    model = FastText(
        sentences=phrases,
        vector_size=vector_size, window=window, min_count=min_count,
        workers=workers, epochs=epochs, seed=seed,
        min_n=min_n, max_n=max_n,
        callbacks=[saver, logger],
    )
    return model
