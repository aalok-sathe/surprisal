from transformers import tokenization_utils_base


def pick_matching_token_ixs(
    encoding: "Encoding",
    char_span_of_interest: slice,
) -> slice:
    """Picks token indices in a tokenized encoded sequence that best correspond to
        a substring of interest in the original sequence, given by a char span (slice)
    Args:
        batchencoding (transformers.tokenization_utils_base.BatchEncoding): the output of a
            `tokenizer(text)` call on a single text instance (not a batch, i.e. `tokenizer([text])`).
        char_span_of_interest (slice): a `slice` object denoting the character indices in the
            original `text` string we want to extract the corresponding tokens for
    Returns:
        slice: the start and stop indices within an encoded sequence that
            best match the `char_span_of_interest`
    """

    start_token = 0
    end_token = len(encoding.ids)
    for i, _ in enumerate(encoding.ids):
        span = encoding.token_to_chars(
            i
        )  # batchencoding 0 gives access to the encoded string

        if span is None:  # for [CLS], no span is returned
            # log(
            #     f'No span returned for token at {i}: "{batchencoding.tokens()[i]}"',
            #     type="WARN",
            #     cmap="WARN",
            #     verbosity_check=True,
            # )
            continue
        else:
            span = tokenization_utils_base.CharSpan(*span)

        if span.start <= char_span_of_interest.start:
            start_token = i
        if span.end >= char_span_of_interest.stop:
            end_token = i + 1
            break

    assert end_token - start_token <= len(
        encoding.ids
    ), f"Extracted span is larger than original span"

    return slice(start_token, end_token)
