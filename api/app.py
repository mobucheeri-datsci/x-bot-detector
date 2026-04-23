import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import re
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.config import checkpoints, device, max_seq_len, data_processed, numeric_features

_model = None
_model_info = None
_tokenizer = None
_numeric_mean = None
_numeric_std = None
_threshold = 0.5

def load_model():
    global _model, _model_info, _tokenizer, _numeric_mean, _numeric_std, _threshold
    if _model is not None:
        return
    proc_path = os.path.join(data_processed, "processed.pt")
    if os.path.exists(proc_path):
        proc_data = torch.load(proc_path, weights_only=False)
        _numeric_mean = proc_data.get("numeric_mean")
        _numeric_std = proc_data.get("numeric_std")
    info_path = os.path.join(checkpoints, "best_model_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError("No trained model. Run: python src/train.py")
    with open(info_path) as f:
        _model_info = json.load(f)
    name = _model_info["model_name"]
    model_type = _model_info.get("model_type", "neural")
    _threshold = float(_model_info.get("threshold", 0.5))
    if model_type == "xgboost":
        import xgboost as xgb
        _model = xgb.XGBClassifier()
        _model.load_model(os.path.join(checkpoints, f"{name}_best.json"))
        _tokenizer = None
    else:
        from src.data import GloveVocab
        _tokenizer = GloveVocab.load(os.path.join(checkpoints, "vocab.json"))
        from src.models import BiGRU_LSTM, CNN_BiLSTM
        _model = BiGRU_LSTM(vocab_size=_tokenizer.vocab_size) if name == "bigru_lstm" else CNN_BiLSTM(vocab_size=_tokenizer.vocab_size)

        ckpt = os.path.join(checkpoints, f"{name}_best.pt")
        _model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        _model.to(device)
        _model.eval()

def prepare_text(profile):
    parts = []
    bio = str(profile.get("bio", "") or profile.get("description", "") or "")
    if bio.strip():
        parts.append(bio.strip())
    for t in (profile.get("recent_tweets", []) or [])[:20]:
        t = str(t).strip()
        if t:
            parts.append(t)
    combined = " [SEP] ".join(parts)
    combined = re.sub(r"http\S+", "<URL>", combined)
    return re.sub(r"\s+", " ", combined).strip() or "<EMPTY>"

def extract_numeric(profile):
    followers = float(profile.get("followers_count", 0))
    friends = float(profile.get("following_count", 0) or profile.get("friends_count", 0))
    statuses = float(profile.get("tweet_count", 0) or profile.get("statuses_count", 0))
    favourites = float(profile.get("favourites_count", 0))
    age = max(float(profile.get("account_age_days", 365)), 1.0)
    tweets_per_day = statuses / age
    bio = str(profile.get("bio", "") or profile.get("description", "") or "")
    username = str(profile.get("username", "") or profile.get("screen_name", "") or "")
    location = str(profile.get("location", "") or "")
    verified = int(profile.get("is_verified", False) or profile.get("verified", False))
    default_profile = int(profile.get("default_profile", False))
    default_avatar = int(profile.get("has_default_avatar", False) or profile.get("default_profile_image", False))
    f2f_ratio = followers / max(friends, 1)
    fav2stat_ratio = favourites / max(statuses, 1)
    fr2fol_ratio = friends / max(followers, 1)
    stat2fol_ratio = statuses / max(followers, 1)
    has_desc = int(len(bio) > 0)
    has_loc = int(len(location) > 0)
    completeness = has_desc + has_loc + (1 - default_profile) + (1 - default_avatar) + verified
    sn_digits = sum(c.isdigit() for c in username)
    sn_digit_ratio = sn_digits / max(len(username), 1)
    sn_underscore = int("_" in username)
    tweets_per_follower = statuses / max(followers, 1)
    tpd_per_follower = tweets_per_day / max(followers, 1)
    bio_urls = len(re.findall(r"http|www\.|\.com|\.net", bio))
    bio_hashtags = bio.count("#")
    bio_mentions = bio.count("@")
    bio_words = len(bio.split()) if bio else 0
    news_pattern = r"\b(?:news|breaking|daily|magazine|journal|times|herald|tribune|gazette|broadcast|media|press|reporter|journalist|editor|anchor|correspondent|coverage|headlines|report)\b"
    org_pattern = r"\b(?:official|corp|inc\.?|llc|ltd|company|brand|store|shop|support|customer|service|team|foundation|organisation|organization|ngo|charity)\b"
    bio_lower = bio.lower()
    bio_has_news = int(bool(re.search(news_pattern, bio_lower)))
    bio_has_org = int(bool(re.search(org_pattern, bio_lower)))
    bio_likely_org = int((bio_has_news or bio_has_org) and followers > 1000 and age > 365)
    is_established = int(bool(verified) and followers > 10000 and age > 365)
    log_followers = float(np.log1p(followers))
    log_friends = float(np.log1p(friends))
    log_statuses = float(np.log1p(statuses))
    log_favourites = float(np.log1p(favourites))
    log_tpf = float(np.log1p(tweets_per_follower))
    log_f2f = float(np.log1p(f2f_ratio))
    return [
        followers, friends, statuses, favourites, age, tweets_per_day,
        log_followers, log_friends, log_statuses, log_favourites, log_tpf, log_f2f,
        f2f_ratio, fav2stat_ratio, fr2fol_ratio, stat2fol_ratio,
        verified, default_profile, default_avatar,
        has_desc, has_loc, completeness, len(bio), len(username),
        sn_digits, sn_digit_ratio, sn_underscore,
        tweets_per_follower, tpd_per_follower,
        bio_urls, bio_hashtags, bio_mentions, bio_words,
        bio_has_news, bio_has_org, bio_likely_org, is_established,
    ]

feature_descriptions = {
    "followers_count": "total followers",
    "friends_count": "total accounts followed",
    "statuses_count": "total tweets posted",
    "favourites_count": "total likes given",
    "account_age_days": "how long the account has existed",
    "average_tweets_per_day": "tweets posted per day on average",
    "log_followers_count": "follower count (log scale)",
    "log_friends_count": "following count (log scale)",
    "log_statuses_count": "tweet count (log scale)",
    "log_favourites_count": "likes given (log scale)",
    "log_tweets_per_follower": "tweets per follower (log scale)",
    "log_followers_to_friends_ratio": "follower-to-following balance (log scale)",
    "followers_to_friends_ratio": "how many followers per account followed",
    "favourites_to_statuses_ratio": "likes given per tweet posted",
    "friends_to_followers_ratio": "how many followed per follower",
    "statuses_to_followers_ratio": "tweets per follower",
    "verified": "has the verified blue checkmark",
    "default_profile": "still using the default profile theme",
    "default_profile_image": "still using the default avatar",
    "has_description": "has filled in a bio",
    "has_location": "has filled in a location",
    "profile_completeness": "how many profile fields are filled in",
    "description_length": "length of the bio",
    "screen_name_length": "length of the username",
    "screen_name_digits": "number of digits in the username",
    "screen_name_digit_ratio": "fraction of the username that is digits",
    "screen_name_has_underscore": "username contains an underscore",
    "tweets_per_follower": "tweets posted per follower",
    "tweets_per_day_per_follower": "tweets per day relative to followers",
    "bio_url_count": "URLs in the bio",
    "bio_hashtag_count": "hashtags in the bio",
    "bio_mention_count": "mentions in the bio",
    "bio_word_count": "words in the bio",
    "bio_has_news_keywords": "bio mentions news or journalism",
    "bio_has_org_keywords": "bio mentions an organisation or brand",
    "bio_likely_organisation": "bio plus reach suggests a real organisation",
    "is_established_account": "verified, large following, account older than one year",
}

def format_feature_value(name, value):
    if name == "verified":
        return "yes" if value > 0.5 else "no"
    if name in ("default_profile", "default_profile_image", "has_description", "has_location",
                "screen_name_has_underscore", "bio_has_news_keywords", "bio_has_org_keywords",
                "bio_likely_organisation", "is_established_account"):
        return "yes" if value > 0.5 else "no"
    if name == "account_age_days":
        years = value / 365.0
        if years >= 1:
            return f"{years:.1f} yrs"
        return f"{int(value)} days"
    if name in ("followers_count", "friends_count", "statuses_count", "favourites_count"):
        if value >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value/1_000:.1f}K"
        return str(int(value))
    if name == "average_tweets_per_day":
        return f"{value:.1f}/day"
    if name == "profile_completeness":
        return f"{int(value)}/5"
    if name == "screen_name_length":
        return f"{int(value)} chars"
    if name.startswith("log_"):
        return f"{value:.2f}"
    if "ratio" in name:
        return f"{value:.2f}"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)

def compute_contributions(numeric_arr, raw_numeric):
    if _model_info.get("model_type") != "xgboost":
        return None
    import xgboost as xgb
    booster = _model.get_booster()
    dmatrix = xgb.DMatrix(numeric_arr.reshape(1, -1))
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]
    feat_contribs = contribs[:-1]
    indexed = sorted(enumerate(feat_contribs), key=lambda x: abs(x[1]), reverse=True)
    total_abs = sum(abs(c) for _, c in indexed if abs(c) >= 0.01)
    toward_bot, toward_human = [], []
    for idx, contrib in indexed:
        if abs(contrib) < 0.01:
            continue
        if len(toward_bot) >= 4 and len(toward_human) >= 4:
            break
        name = numeric_features[idx]
        entry = {
            "feature": name,
            "description": feature_descriptions.get(name, name.replace("_", " ")),
            "value": format_feature_value(name, float(raw_numeric[idx])),
            "contribution": round(float(contrib), 3),
            "percentage": round(float(abs(contrib) / max(total_abs, 0.001)) * 100, 1),
        }
        if contrib > 0 and len(toward_bot) < 4:
            toward_bot.append(entry)
        elif contrib < 0 and len(toward_human) < 4:
            toward_human.append(entry)
    return {"toward_bot": toward_bot, "toward_human": toward_human}

def generate_signals(profile, score):
    signals = []
    followers = int(profile.get("followers_count", 0))
    following = int(profile.get("following_count", 0) or profile.get("friends_count", 0))
    tweets = int(profile.get("tweet_count", 0) or profile.get("statuses_count", 0))
    age = max(int(profile.get("account_age_days", 365)), 1)
    if followers / max(following, 1) < 0.1 and following > 100:
        signals.append("Very low follower-to-following ratio")
    if age < 30:
        signals.append("Account is less than 30 days old")
    if tweets / age > 50:
        signals.append("Extremely high tweet frequency")
    if profile.get("has_default_avatar", False) or profile.get("default_profile_image", False):
        signals.append("Using default profile image")
    if followers < 5 and following > 500:
        signals.append("Mass-following with few followers")
    if len(str(profile.get("bio", "") or "")) < 5:
        signals.append("Empty or very short bio")
    if not signals and score >= 70:
        signals.append("Text patterns indicate automated content")
    if not signals:
        signals.append("No strong bot signals detected")
    return signals

def predict(profile):
    load_model()
    raw_numeric = extract_numeric(profile)
    numeric_arr = np.array(raw_numeric, dtype=np.float32)
    if _numeric_mean is not None and _numeric_std is not None:
        numeric_arr = (numeric_arr - _numeric_mean) / _numeric_std
    name = _model_info["model_name"]
    model_type = _model_info.get("model_type", "neural")
    if model_type == "xgboost":
        bot_prob = float(_model.predict_proba(numeric_arr.reshape(1, -1))[0, 1])
    else:
        text = prepare_text(profile)
        numeric = torch.tensor([numeric_arr], dtype=torch.float32, device=device)
        with torch.no_grad():
            tokens = _tokenizer.tokenize_batch([text], max_len=max_seq_len).to(device)
            logits = _model(input_ids=tokens, numeric=numeric)
            bot_prob = torch.sigmoid(logits.squeeze()).item()
    raw_followers, raw_age = raw_numeric[0], raw_numeric[4]
    raw_verified, raw_likely_org = raw_numeric[16], raw_numeric[35]
    override_applied = None
    if raw_likely_org and raw_verified and raw_age > 365 and raw_followers > 10_000:
        capped = max(0.0, _threshold - 0.15)
        if bot_prob > capped:
            override_applied = "news_org"
        bot_prob = min(bot_prob, capped)
    score = int(round(bot_prob * 100))
    margin = 0.18 if raw_age < 60 else 0.1
    delta = bot_prob - _threshold
    if abs(delta) <= margin:
        label = "uncertain"
    elif delta > 0:
        label = "bot"
    else:
        label = "human"
    return {
        "username": profile.get("username", ""),
        "bot_probability": round(bot_prob, 4),
        "bot_score": score,
        "label": label,
        "confidence": "high" if abs(delta) > 0.3 else ("medium" if abs(delta) > 0.15 else "low"),
        "signals": generate_signals(profile, score),
        "contributions": compute_contributions(numeric_arr, raw_numeric),
        "override_applied": override_applied,
        "threshold": round(_threshold, 4),
        "margin": round(margin, 4),
    }

class PredictRequest(BaseModel):
    username: str
    display_name: str = ""
    bio: str = ""
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    listed_count: int = 0
    account_age_days: int = 365
    recent_tweets: list[str] = []
    has_default_avatar: bool = False
    is_verified: bool = False
    url: str = ""

class PredictResponse(BaseModel):
    username: str
    bot_probability: float
    bot_score: int
    label: str
    confidence: str
    signals: list[str]
    contributions: dict | None = None
    override_applied: str | None = None
    threshold: float = 0.5
    margin: float = 0.1

app = FastAPI(title="Twitter Bot Detector API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(https://(x|twitter)\.com|chrome-extension://.*)$",
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

@app.on_event("startup")
async def startup():
    try:
        load_model()
        print("[+] Model loaded")
    except FileNotFoundError:
        print("[!] No model found, train first with: python src/train.py")
    except Exception as e:
        print(f"[!] Model load failed: {e}")

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    try:
        return PredictResponse(**predict(request.model_dump()))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BatchRequest(BaseModel):
    profiles: list[PredictRequest]

class BatchResponse(BaseModel):
    results: list[PredictResponse]

@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch_endpoint(request: BatchRequest):
    if len(request.profiles) > 50:
        raise HTTPException(status_code=429, detail="batch limit is 50 profiles")
    try:
        results = [PredictResponse(**predict(p.model_dump())) for p in request.profiles]
        return BatchResponse(results=results)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ThreadReplyRequest(BaseModel):
    username: str
    display_name: str = ""
    is_verified: bool = False
class ThreadReplyResponse(BaseModel):
    username: str
    flag: str
    reasons: list[str]
class ThreadReplyBatchRequest(BaseModel):
    replies: list[ThreadReplyRequest]
class ThreadReplyBatchResponse(BaseModel):
    results: list[ThreadReplyResponse]
def score_thread_reply(profile):
    username = profile.get("username", "")
    is_verified = profile.get("is_verified", False)
    if is_verified:
        return {
            "username": username,
            "flag": "typical",
            "reasons": ["verified account"],
        }
    signals = 0
    reasons = []
    digits = sum(c.isdigit() for c in username)
    if digits >= 5:
        signals += 2
        reasons.append(f"username contains {digits} digits")
    elif digits >= 3:
        signals += 1
        reasons.append(f"username contains {digits} digits")
    if re.search(r"\d{4,}$", username):
        signals += 1
        reasons.append("username ends in long digit sequence")
    if len(username) >= 12 and digits / max(len(username), 1) > 0.3:
        signals += 1
        reasons.append("handle is mostly digits")
    if re.match(r"^[a-z]+\d+$", username.lower()):
        signals += 1
        reasons.append("handle follows auto-generated pattern")
    if signals >= 3:
        flag = "suspicious"
    elif signals >= 1:
        flag = "possibly_suspicious"
    else:
        flag = "typical"
        reasons = ["no obvious red flags in visible info"]
    return {"username": username, "flag": flag, "reasons": reasons}

@app.post("/predict_thread_batch", response_model=ThreadReplyBatchResponse)
async def predict_thread_batch_endpoint(request: ThreadReplyBatchRequest):
    if len(request.replies) > 100:
        raise HTTPException(status_code=429, detail="batch limit is 100 replies")
    results = [ThreadReplyResponse(**score_thread_reply(r.model_dump())) for r in request.replies]
    return ThreadReplyBatchResponse(results=results)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_name": _model_info.get("model_name", "") if _model_info else "",
    }