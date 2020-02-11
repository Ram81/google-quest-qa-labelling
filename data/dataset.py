from imports import *
from params import *

from data.text_cleaning import *
from data.text_converting import *


class QATrainDataset(Dataset):
    def __init__(
        self,
        df,
        transformer,
        max_len_q=200,
        max_len_a=200,
        max_len_t=60,
        augment=False,
        special=False,
    ):
        super().__init__()
        self.tokens = []
        self.ids = []

        df = df.copy()
        df["question_title"] = df["question_title"].apply(clean_text)
        df["question_body"] = df["question_body"].apply(clean_text)
        df["answer"] = df["answer"].apply(clean_text)

        for title, question, answer in zip(
            df["question_title"].values, df["question_body"].values, df["answer"].values
        ):
            if special:
                tokens, idx, position, q_pos, a_pos = convert_text_special(
                    title,
                    question,
                    answer,
                    transformer,
                    augment=augment,
                    max_len_q=max_len_q,
                    max_len_a=max_len_a,
                    max_len_t=max_len_t,
                )
            else:
                tokens, idx, position, q_pos, a_pos = convert_text(
                    title,
                    question,
                    answer,
                    transformer,
                    augment=augment,
                    max_len_q=max_len_q,
                    max_len_a=max_len_a,
                    max_len_t=max_len_t,
                )

            self.tokens.append(tokens)
            self.ids.append(idx)

        self.tokens = np.array(self.tokens)
        self.ids = np.array(self.ids)
        self.df = df 

        self.y = np.array(df[TARGETS])
        self.y = (self.y - YMIN) / (YMAX - YMIN)


    def __len__(self):
        return len(self.df)

    def getembed(self, idx):
        row = self.df.iloc[idx]

        host = HOST_EMB_LIST.get(row.host)
        cat = CAT_EMB_LIST.get(row.category)

        if host is None:
            host = HOST_EMB_LIST.get("unknown")

        if cat is None:
            cat = CAT_EMB_LIST.get("unknown")
            
        return host, cat

    def __getitem__(self, idx):

        host, cat = self.getembed(idx)

        return (
            torch.tensor(self.tokens[idx]),
            torch.tensor(self.ids[idx]),
            torch.tensor(host),
            torch.tensor(cat),
            torch.tensor(self.y[idx]),
        )


class QATrainDatasetSep(Dataset):
    """
    Question and answer are separed
    """
    def __init__(
        self,
        df,
        transformer,
        max_len_q=200,
        max_len_a=200,
        max_len_t=60,
        special=False,
    ):
        super().__init__()
        self.df = df

        self.tokens_q = []
        self.tokens_a = []
        self.idxs_q = []
        self.idxs_a = []

        for title, question, answer in zip(
            df["question_title"].values, df["question_body"].values, df["answer"].values
        ):

            tokens_q, tokens_a, idx_q, idx_a = convert_text_sep(
                title,
                question,
                answer,
                transformer,
                max_len_q=512,
                max_len_a=512,
                max_len_t=max_len_t,
                use_special=special,
            )

            self.tokens_q.append(tokens_q)
            self.tokens_a.append(tokens_a)
            self.idxs_q.append(idx_q)
            self.idxs_a.append(idx_a)

        self.tokens_q = np.array(self.tokens_q)
        self.tokens_a = np.array(self.tokens_a)
        self.idxs_q = np.array(self.idxs_q)
        self.idxs_a = np.array(self.idxs_a)

        self.y = np.array(df[TARGETS])
        self.y = (self.y - YMIN) / (YMAX - YMIN)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.tokens_q[idx]),
            torch.tensor(self.tokens_a[idx]),
            torch.tensor(self.idxs_q[idx]),
            torch.tensor(self.idxs_a[idx]),
            torch.tensor(self.y[idx]),
        )


class QATestDataset(Dataset):
    def __init__(
        self, df, transformer, max_len_q=200, max_len_a=200, max_len_t=60, special=False
    ):
        super().__init__()
        self.df = df

        self.tokens = []
        self.question = []
        self.positions = []
        self.q_idx = []
        self.a_idx = []

        for title, question, answer in zip(
            df["question_title"].values, df["question_body"].values, df["answer"].values
        ):
            if special:
                tokens, idx, position, q_pos, a_pos = convert_text_special(
                    title,
                    question,
                    answer,
                    transformer,
                    max_len_q=max_len_q,
                    max_len_a=max_len_a,
                    max_len_t=max_len_t,
                )
            else:
                tokens, idx, position, q_pos, a_pos = convert_text(
                    title,
                    question,
                    answer,
                    transformer,
                    max_len_q=max_len_q,
                    max_len_a=max_len_a,
                    max_len_t=max_len_t,
                )
            self.tokens.append(tokens)
            self.question.append(idx)
            self.positions.append(position)
            self.q_idx.append(q_pos)
            self.a_idx.append(a_pos)

        self.tokens = np.array(self.tokens)
        self.question = np.array(self.question)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.tokens[idx]),
            torch.tensor(self.question[idx]),
            torch.tensor(self.positions[idx]),
            torch.tensor(self.q_idx[idx]),
            torch.tensor(self.a_idx[idx]),
            torch.tensor(0),
        )

