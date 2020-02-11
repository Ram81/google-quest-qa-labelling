from imports import *
from params import *


def trim_input(t, q, a, max_len_q=200, max_len_a=200, max_len_t=60):
    max_len = max_len_t + max_len_q + max_len_a + 4
    len_t, len_q, len_a = len(t), len(q), len(a)

    if max_len_t > len_t:
        new_len_t = len_t
        max_len_a = max_len_a + floor((max_len_t - len_t) / 2)
        max_len_q = max_len_q + ceil((max_len_t - len_t) / 2)
    else:
        new_len_t = max_len_t

    if max_len_a > len_a:
        new_len_a = len_a
        new_len_q = max_len_q + (max_len_a - len_a)
    elif max_len_q > len_q:
        new_len_a = max_len_a + (max_len_q - len_q)
        new_len_q = len_q
    else:
        new_len_a = max_len_a
        new_len_q = max_len_q

    return t[:new_len_t], q[:new_len_q], a[:new_len_a]


def augment_text(text, min_len=50, kept_prop=0.9):
    if kept_prop * len(text) > min_len:
        # kept_prop =  min_len / len(text)
        mask = np.random.random(len(text)) < kept_prop
        return list(np.array(text)[mask])
    else:
        return text


def truncate_sequences(question, answer, max_length=512, stride=0):
    num_tokens_to_remove = len(question) + len(answer) - max_length

    overflowing_tokens = []
    for _ in range(num_tokens_to_remove):
        if len(question) > len(answer):
            overflowing_tokens = [question[-1]] + overflowing_tokens
            question = question[:-1]
        else:
            answer = answer[:-1]
    window_len = min(len(question), stride)
    if window_len > 0:
        overflowing_tokens = question[-window_len:] + overflowing_tokens

    return question, answer, overflowing_tokens


def convert_text2(
    title,
    question,
    answer,
    transformer,
    max_len_q=200,
    max_len_a=200,
    max_len_t=50,
    augment=False,
    margin=20,
):
    max_len = 4 + max_len_q + max_len_a + max_len_t  # with sep tokens

    tokens_t = transformer.tokenizer.tokenize(title)

    question = transformer.tokenizer.tokenize(question)
    answer = transformer.tokenizer.tokenize(answer)

    if augment:
        question = augment_text(question, min_len=max_len_q + margin)
        answer = augment_text(answer, min_len=max_len_a + margin)

    tokens_q, tokens_a, _ = truncate_sequences(
        question, answer, max_length=max_len - len(tokens_t) - 4
    )

    tokens = ["[CLS]"] + tokens_t + ["[q]"] + tokens_q + ["[a]"] + tokens_a + ["[SEP]"]

    q_pos = 1 + len(tokens_t)
    a_pos = q_pos + len(tokens_q) + 1

    assert tokens[q_pos] == "[q]", tokens[q_pos]
    assert tokens[a_pos] == "[a]", tokens[a_pos]

    question = transformer.tokenizer.convert_tokens_to_ids(tokens)
    segments = (
        [0] * (1 + len(tokens_t))
        + [1] * (1 + len(tokens_q))
        + [2] * (2 + len(tokens_a))
    )

    positions = [i for i in range(len(tokens_t) + len(tokens_q) + 3)] + [
        i for i in range(max_len)
    ]

    padding = [0] * (max_len - len(question))

    return question + padding, segments + padding, positions[:max_len], q_pos, a_pos


def convert_text(
    title,
    question,
    answer,
    transformer,
    max_len_q=200,
    max_len_a=200,
    max_len_t=50,
    augment=False,
    margin=20,
):
    max_len = 4 + max_len_q + max_len_a + max_len_t  # with sep tokens

    title = transformer.tokenizer.tokenize(title)
    question = transformer.tokenizer.tokenize(question)
    answer = transformer.tokenizer.tokenize(answer)

    if augment:
        question = augment_text(question, min_len=max_len_q + margin)
        answer = augment_text(answer, min_len=max_len_a + margin)

    tokens_t, tokens_q, tokens_a = trim_input(
        title,
        question,
        answer,
        max_len_t=max_len_t,
        max_len_q=max_len_q,
        max_len_a=max_len_a,
    )

    tokens = ["[CLS]"] + tokens_t + ["[q]"] + tokens_q + ["[a]"] + tokens_a + ["[SEP]"]

    q_pos = 1 + len(tokens_t)
    a_pos = q_pos + len(tokens_q) + 1

    assert tokens[q_pos] == "[q]", tokens[q_pos]
    assert tokens[a_pos] == "[a]", tokens[a_pos]

    # tokens = ["[CLS]"] + tokens_t + ["[SEP]"] + tokens_q + ["[SEP]"] + tokens_a + ["[SEP]"]

    question = transformer.tokenizer.convert_tokens_to_ids(tokens)

    # segments = [0] * (3 + len(tokens_t) + len(tokens_q)) + [1] * (1 + len(tokens_a))
    segments = (
        [0] * (1 + len(tokens_t))
        + [1] * (1 + len(tokens_q))
        + [2] * (2 + len(tokens_a))
    )

    positions = [i for i in range(len(tokens_t) + len(tokens_q) + 3)] + [
        i for i in range(max_len)
    ]
    # positions = [i for i in range(len(tokens_t) + len(tokens_q) + 3)] +  [i for i in range(max_len_t + max_len_q + 3, max_len)] + [0] * 1000

    padding = [0] * (max_len - len(question))

    return question + padding, segments + padding, positions[:max_len], q_pos, a_pos


def convert_text_sep(
    title,
    question,
    answer,
    transformer,
    max_len_q=512,
    max_len_a=512,
    max_len_t=50,
    use_special=False,
):

    title = transformer.tokenizer.tokenize(title)
    question = transformer.tokenizer.tokenize(question)
    answer = transformer.tokenizer.tokenize(answer)

    tokens_t = title[:max_len_t]

    if use_special:
        tokens_q = ["[q]"] + tokens_t + ["[SEP]"] + question
        tokens_a = ["[a]"] + tokens_t + ["[SEP]"] + answer
    else:
        tokens_q = ["[CLS]"] + tokens_t + ["[SEP]"] + question
        tokens_a = ["[CLS]"] + tokens_t + ["[SEP]"] + answer

    tokens_q = tokens_q[: max_len_q - 1] + ["[SEP]"]
    tokens_a = tokens_a[: max_len_a - 1] + ["[SEP]"]

    question = transformer.tokenizer.convert_tokens_to_ids(tokens_q)
    answer = transformer.tokenizer.convert_tokens_to_ids(tokens_a)

    if use_special:
        segments_q = [0] * (1 + len(tokens_t)) + [1] * (
            len(question) - (1 + len(tokens_t))
        )
        segments_a = [0] * (1 + len(tokens_t)) + [2] * (
            len(answer) - (1 + len(tokens_t))
        )
    else:
        segments_q = [0] * (1 + len(tokens_t)) + [1] * (
            len(question) - (1 + len(tokens_t))
        )
        segments_a = [0] * (1 + len(tokens_t)) + [1] * (
            len(answer) - (1 + len(tokens_t))
        )

    padding_q = [0] * (max_len_q - len(question))
    padding_a = [0] * (max_len_a - len(answer))

    return (
        question + padding_q,
        answer + padding_a,
        segments_q + padding_q,
        segments_a + padding_a,
    )


def convert_text_special(
    title,
    question,
    answer,
    transformer,
    max_len_q=200,
    max_len_a=200,
    max_len_t=50,
    augment=False,
    margin=20,
):
    max_len_q -= len(SPECIAL_TOKENS) - 1
    max_len = 33 + max_len_q + max_len_a + max_len_t

    title = transformer.tokenizer.tokenize(title)
    question = transformer.tokenizer.tokenize(question)
    answer = transformer.tokenizer.tokenize(answer)

    if augment:
        question = augment_text(question, min_len=max_len_q + margin)
        answer = augment_text(answer, min_len=max_len_a + margin)

    tokens_t, tokens_q, tokens_a = trim_input(
        title,
        question,
        answer,
        max_len_t=max_len_t,
        max_len_q=max_len_q,
        max_len_a=max_len_a,
    )

    tokens = (
        SPECIAL_TOKENS + tokens_t + ["[q]"] + tokens_q + ["[a]"] + tokens_a + ["[SEP]"]
    )

    question = transformer.tokenizer.convert_tokens_to_ids(tokens)

    segments = (
        [0] * (len(SPECIAL_TOKENS) + len(tokens_t))
        + [1] * (1 + len(tokens_q))
        + [2] * (2 + len(tokens_a))
    )

    positions = [i for i in range(len(tokens_t) + len(tokens_q) + 3)] + [
        i for i in range(max_len)
    ]

    padding = [0] * (max_len - len(question))

    return question + padding, segments + padding, positions[:max_len], 0, 0

