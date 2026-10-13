
def precision_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int = 3,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")

    return len(set(retrieved[:k]) & set(relevant)) / k


def recall_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int = 3,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")

    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    return len(set(retrieved[:k]) & relevant_set) / len(relevant_set)


def reciprocal_rank(
    retrieved: list[str],
    relevant: list[str],
) -> float:
    relevant_set = set(relevant)

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0
