from util import *
from imports import *

from training.radam import *
from training.freeze import *
from training.sampler import *
from training.learning_rate import *


def predict(model, dataset, batch_size=8, sep=False):
    model.eval()
    preds = np.empty((0, NUM_TARGETS))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    with torch.no_grad():
        for data in loader:
            if sep:
                tokens_q, tokens_a, idx_q, idx_a, y_batch = data
                y_pred = model(tokens_q.cuda(), tokens_a.cuda(), idx_q.cuda(), idx_a.cuda()).detach()
            else:
                tokens, idx, host, cat, y_batch = data
                y_pred = model(tokens.cuda(), idx.cuda(), host.cuda(), cat.cuda()).detach()

            preds = np.concatenate([preds, torch.sigmoid(y_pred).cpu().numpy()])

    del y_pred, y_batch, loader
    torch.cuda.empty_cache()

    return preds