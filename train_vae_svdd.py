import numpy as np
from scripts.vae_svdd_trainer import Trainer

X_train = np.load("./data/in/X.npy")
y_train = np.load("./data/in/Y.npy")

trainer = Trainer(X_train, y_train)
trainer.fit()
