import autograd.numpy as np
from sklearn.preprocessing import LabelEncoder


def prepare_vectors(winners, losers, days_since_start, covariates=None):

    encoder = LabelEncoder()
    encoder.fit(winners.tolist() + losers.tolist())
    n_players = len(encoder.classes_)

    # Let's put the big arrays together.
    n_matches = winners.shape[0]

    # Starts and ends for players
    s = np.zeros(n_players, dtype=int)
    e = np.zeros(n_players, dtype=int)

    # Lookups for matches
    w = np.zeros(n_matches, dtype=int)
    l = np.zeros(n_matches, dtype=int)  # NOQA

    # Features (days since)
    X = np.zeros(2*n_matches)

    if covariates is not None:
        cov_array = np.zeros((2*n_matches, covariates.shape[1]))
    else:
        cov_array = None

    matches_seen = 0

    for i, cur_player in enumerate(encoder.classes_):

        relevant = (winners == cur_player) | (losers == cur_player)
        relevant_indices = np.where(relevant)[0]

        relevant_winners = winners[relevant]
        relevant_losers = losers[relevant]
        relevant_days = days_since_start[relevant]

        s[i] = matches_seen
        e[i] = matches_seen + np.sum(relevant).astype(int)

        for j, (cur_winner, cur_loser, cur_days, cur_index) in enumerate(zip(
                relevant_winners, relevant_losers, relevant_days,
                relevant_indices)):

            if cur_winner == cur_player:
                w[cur_index] = s[i] + j
            else:
                l[cur_index] = s[i] + j

        if covariates is not None:
            relevant_covs = covariates[relevant]
            cov_array[s[i]:e[i], :] = relevant_covs
        else:
            relevant_covs = None

        X[s[i]:e[i]] = relevant_days
        matches_seen = e[i]

    return encoder, s, e, w, l, X, cov_array
