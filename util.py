from imports import *
from fastai.vision import *
from params import *

import hashlib

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def save_model_weights(model, filename, verbose=1, cp_folder=CP_PATH):
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def load_model_weights(model, filename, verbose=1, cp_folder=CP_PATH, strict=True):
    if verbose:
        print(f"\n -> Loading weights from {os.path.join(cp_folder, filename)}\n")
    try:
        model.load_state_dict(os.path.join(cp_folder, filename), strict=strict)
    except BaseException:
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=strict,
        )
    return model


def count_parameters(model, all=False):
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert isinstance(size, torch.Size)
    return " × ".join(map(str, size))


def dump_tensors(gpu_only=True, verbose=0):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    if verbose:
                        print(
                            "%s:%s%s %s"
                            % (
                                type(obj).__name__,
                                " GPU" if obj.is_cuda else "",
                                " pinned" if obj.is_pinned else "",
                                pretty_size(obj.size()),
                            )
                        )
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    if verbose:
                        print(
                            "%s → %s:%s%s%s%s %s"
                            % (
                                type(obj).__name__,
                                type(obj.data).__name__,
                                " GPU" if obj.is_cuda else "",
                                " pinned" if obj.data.is_pinned else "",
                                " grad" if obj.requires_grad else "",
                                " volatile" if obj.volatile else "",
                                pretty_size(obj.data.size()),
                            )
                        )
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", total_size)


def calc_hash( mystr ):
    return int( hashlib.md5( mystr.encode('utf-8') ).hexdigest(), 16 )


class CustomGroupKFold():
    def __init__(self, n_splits, random_state=2020):
        self.n_splits = n_splits
        self.random_state = random_state
        self.group = None

    def split(self, X=None, y=None, groups=None):
        seed = '_' + str(self.random_state)
        self.group = groups.values.copy()
        indexes = np.array([calc_hash(str(val) + seed) % self.n_splits for val in self.group])
        for fold in range(self.n_splits):
            train_idx = np.where(indexes != fold)[0]
            test_idx = np.where(indexes == fold)[0]
            yield train_idx, test_idx