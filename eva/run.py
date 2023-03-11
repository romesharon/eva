import numpy as np


def generate_y_true(true_samples=2000, false_samples=500) -> np.ndarray:
    true = np.random.randint(1, 2, size=true_samples)
    false = np.random.randint(1, size=false_samples)
    y_true = np.concatenate((false, true), axis=None)
    np.random.shuffle(y_true)
    return y_true


def generate_y_pred(y_true: np.ndarray) -> np.ndarray:
    true = np.random.randint(1, 2, size=int(y_true.size * 9))
    false = np.random.randint(1, size=int(y_true.size *  10000))
    choice = np.concatenate((false, true), axis=None)
    np.random.shuffle(choice)
    y_pred = np.array(y_true)
    y_pred = [np.random.choice(choice, 1)[0] if y == 1 else np.random.choice(choice, 1)[0] for y in y_pred]
    return y_pred


# y_true = generate_y_true()
# y_pred = generate_y_pred(y_true)
# y_prob = np.zeros(len(y_true))
#
# eva = Eva(y_true, y_pred, y_prob)
# eva.evaluate()


from eva.eva import Eva

import numpy as np

# Generate y_true list with 100 elements
y_true = np.random.randint(0, 2, size=1000)

# Simulate y_pred for a bad model with 70% accuracy
y_pred = []
for i in range(len(y_true)):
    if np.random.rand() < 0.5:
        y_pred.append(y_true[i])
    else:
        y_pred.append(1 - y_true[i])

# Simulate y_prob for a bad model with low confidence
y_prob = []
for i in range(len(y_true)):
    if y_true[i] == 1:
        y_prob.append(np.random.uniform(0.3, 1))
    else:
        y_prob.append(np.random.uniform(0, 0.7))

# Create an Eva object and evaluate the model
eva = Eva(y_true, y_pred, y_prob)
eva.evaluate()