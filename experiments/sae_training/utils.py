def name_embedding_training(
    name: str, n_hidden: int, n_features: list[int], history: bool = False
):
    return (
        f"{name}_hidden_{n_hidden}_features_{n_features}{'_history' if history else ''}"
    )


def name_sae_evaluation(
    name: str,
    n_hidden: int,
    n_features: list[int],
    sae_labels: list[str],
    l1_coefficients: list[float] | None = None,
):
    return f"{name}_hidden_{n_hidden}_features_{n_features}_saes_{sae_labels}{f'_l1_{l1_coefficients}' if l1_coefficients else ''}"
