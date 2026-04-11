import os
import torch

root_dir       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_raw       = os.path.join(root_dir, "data", "raw")
data_processed = os.path.join(root_dir, "data", "processed")
checkpoints    = os.path.join(root_dir, "models", "checkpoints")

dataset_csv = os.path.join(data_raw, "twitter_human_bots.csv")

glove_dir  = os.path.join(data_raw, "glove")
glove_file = os.path.join(glove_dir, "glove.twitter.27B.200d.txt")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()

numeric_features = [
    "followers_count", "friends_count", "statuses_count", "favourites_count",
    "account_age_days", "average_tweets_per_day",
    "log_followers_count", "log_friends_count", "log_statuses_count",
    "log_favourites_count", "log_tweets_per_follower", "log_followers_to_friends_ratio",
    "followers_to_friends_ratio", "favourites_to_statuses_ratio",
    "friends_to_followers_ratio", "statuses_to_followers_ratio",
    "verified", "default_profile", "default_profile_image",
    "has_description", "has_location", "profile_completeness",
    "description_length", "screen_name_length",
    "screen_name_digits", "screen_name_digit_ratio", "screen_name_has_underscore",
    "tweets_per_follower", "tweets_per_day_per_follower",
    "bio_url_count", "bio_hashtag_count", "bio_mention_count", "bio_word_count",
    "bio_has_news_keywords", "bio_has_org_keywords", "bio_likely_organisation",
    "is_established_account",
]
num_numeric_features = len(numeric_features)

batch_size    = 64
max_epochs    = 50
patience      = 7
learning_rate = 1e-3
weight_decay  = 1e-5
dropout       = 0.3

max_seq_len    = 128
max_vocab_size = 50_000
glove_dim      = 200

cnn_filters     = [3, 4, 5]
cnn_num_filters = 100

rnn_hidden = 128
rnn_layers = 1

label_map     = {"human": 0, "bot": 1}
inv_label_map = {0: "human", 1: "bot"}
