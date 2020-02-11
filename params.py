from util import *
from imports import *

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

SEED = 2019

CP_PATH = f'checkpoints/{date.today()}/'
if not os.path.exists(CP_PATH):
    os.mkdir(CP_PATH)


DATA_PATH = "google-quest-challenge/"

sub = pd.read_csv(DATA_PATH + "sample_submission.csv")
TARGETS = list(sub.columns[1:])
NUM_TARGETS = len(TARGETS)

QUESTION_TARGETS = TARGETS[:21]
ANSWER_DEP_TARGETS = TARGETS[21:26]
ANSWER_INDE_TARGETS  = TARGETS[26:]


NUM_WORKERS = 4
VAL_BS = 8

# from model_zoo.bert import *

TRANSFORMERS  = {
    'bert-base-uncased' : (BertModel,       BertTokenizer,       'bert-base-uncased'),
    'bert-base-cased' : (BertModel,       BertTokenizer,       'bert-base-cased'),
    'bert-large-uncased': (BertModel,       BertTokenizer,       'bert-large-uncased'),
    'bert-large-uncased-whole-word-masking': (BertModel,       BertTokenizer,       'bert-large-uncased-whole-word-masking'),
    'bert-large-uncased-whole-word-masking-finetuned-squad': (BertModel, BertTokenizer, 'bert-large-uncased-whole-word-masking-finetuned-squad'),
    'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
    'roberta-base': (RobertaModel,    RobertaTokenizer,    'roberta-base'),
}


MAX_LEN_QT = 269
MAX_LEN_T = 50
MAX_LEN_Q = 229
MAX_LEN_A = 229

# assert MAX_LEN_Q  + MAX_LEN_A + MAX_LEN_T <= 512 - 4, 'Texts too long for Bert'

df_train = pd.read_csv(DATA_PATH + "train.csv")
y = df_train[TARGETS].values
YMIN = y.min(0)
YMAX = y.max(0)

y = (y - YMIN) / (YMAX - YMIN)
YMEAN = torch.tensor(y.mean(0)[np.newaxis, :])


SPECIAL_TOKENS = [f"[tgt{i}]" for i in range(len(TARGETS))]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def create_emb_list(df, varname):
    cat = {"unknown": 0}
    unique_vals = df[varname].unique()

    for i in range(len(unique_vals)):
        cat[unique_vals[i]] = i + 1

    return cat

HOST_EMB_LIST = create_emb_list(df_train, "host")
CAT_EMB_LIST = create_emb_list(df_train, "category")
