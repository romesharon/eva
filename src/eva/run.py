import numpy as np
from src.eva.eva import Eva


def generate_y_true(true_samples=20000, false_samples=3000) -> np.ndarray:
    true = np.random.randint(1, 2, size=true_samples)
    false = np.random.randint(1, size=false_samples)
    y_true = np.concatenate((false, true), axis=None)
    np.random.shuffle(y_true)
    return y_true


def generate_y_pred(y_true: np.ndarray) -> np.ndarray:
    true = np.random.randint(1, 2, size=int(y_true.size * 0.4))
    false = np.random.randint(1, size=int(y_true.size * 0.3))
    choice = np.concatenate((false, true), axis=None)
    np.random.shuffle(choice)
    y_pred = np.array(y_true)
    y_pred = [np.random.choice(choice, 1)[0] if y == 1 else np.random.choice(choice, 1)[0] for y in y_pred]
    return y_pred


y_true = generate_y_true()
y_pred = generate_y_pred(y_true)
y_prob = np.zeros(len(y_true))

eva = Eva(y_true, y_pred, y_prob)
eva.evaluate()
