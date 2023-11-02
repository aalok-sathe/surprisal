import tokenizers
from transformers import tokenization_utils_base


def hf_pick_matching_token_ixs(
    encoding: "tokenizers.Encoding", span_of_interest: slice, span_type: str
) -> slice:
    """Picks token indices in a tokenized encoded sequence that best correspond to
        a substring of interest in the original sequence, given by a char span (slice)
    Args:
        encoding (transformers.tokenization_utils_base.BatchEncoding): the output of a
            `tokenizer(text)` call on a single text instance (not a batch, i.e. `tokenizer([text])`).
        span_of_interest (slice): a `slice` object denoting the character indices in the
            original `text` string we want to extract the corresponding tokens for
        span_type (str): either `char` or `word`, denoting what type of span we are interested
            in obtaining. this argument has no default to ensure the user is aware of what
            kind of span they are getting from this function
    Returns:
        slice: the start and stop indices of **tokens** within an encoded sequence that
            best match the `span_of_interest`
    """
    span_of_interest = slice(
        span_of_interest.start or 0,
        span_of_interest.stop or len(encoding.ids),
        span_of_interest.step,
    )

    start_token = 0
    end_token = len(encoding.ids)
    for i, _ in enumerate(encoding.ids):
        span = encoding.token_to_chars(i)
        word = encoding.token_to_word(i)
        # batchencoding 0 gives access to the encoded string

        if span is None or word is None:
            # for [CLS], no span is returned
            # log(f'No span returned for token at {i}: "{batchencoding.tokens()[i]}"',
            #      type="WARN", cmap="WARN", verbosity_check=True)
            continue
        else:
            span = tokenization_utils_base.CharSpan(*span)

        if span_type == "char":
            if span.start <= span_of_interest.start:
                start_token = i
            if span.end >= span_of_interest.stop:
                end_token = i + 1
                break
        elif span_type == "word":
            if word < span_of_interest.start:
                start_token = i + 1
            # watch out for the semantics of the "stop"
            if word == span_of_interest.stop:
                end_token = i
            elif word > span_of_interest.stop:
                break

    assert end_token - start_token <= len(
        encoding.ids
    ), f"Extracted span is larger than original span"

    return slice(start_token, end_token)


openai_models_list = [
    "davinci-instruct-beta",
    "babbage",
    "text-similarity-ada-001",
    "babbage-code-search-code",
    "code-davinci-edit-001",
    "ada",
    "ada-similarity",
    "babbage-search-query",
    "text-search-curie-query-001",
    "babbage-search-document",
    "davinci-search-document",
    "text-curie-001",
    "text-similarity-babbage-001",
    "text-similarity-curie-001",
    "code-search-ada-text-001",
    "text-search-ada-doc-001",
    "audio-transcribe-001",
    "text-search-curie-doc-001",
    "curie-similarity",
    "ada-search-document",
    "text-davinci-insert-001",
    "text-search-davinci-doc-001",
    "ada-search-query",
    "text-search-ada-query-001",
    "text-davinci-001",
    "curie",
    "curie-instruct-beta",
    "babbage-similarity",
    "ada-code-search-text",
    "davinci-similarity",
    "text-search-davinci-query-001",
    "babbage-code-search-text",
    "code-search-babbage-code-001",
    "text-davinci-002",
    "text-davinci-003",
    "text-ada-001",
    "davinci-search-query",
    "ada-code-search-code",
    "curie-search-document",
    "text-similarity-davinci-001",
    "text-davinci-insert-002",
    "code-search-babbage-text-001",
    "text-davinci-edit-001",
    "text-search-babbage-query-001",
    "davinci",
    "text-search-babbage-doc-001",
    "curie-search-query",
    "text-babbage-001",
    "code-search-ada-code-001",
    "cushman:2020-05-03",
    "ada:2020-05-03",
    "babbage:2020-05-03",
    "curie:2020-05-03",
    "davinci:2020-05-03",
    "if-davinci-v2",
    "if-curie-v2",
    "if-davinci:3.0.0",
    "davinci-if:3.0.0",
    "davinci-instruct-beta:2.0.0",
    "text-ada:001",
    "text-davinci:001",
    "text-curie:001",
    "text-babbage:001",
]
