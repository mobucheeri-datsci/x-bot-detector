import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import checkpoints

orgs = [
    ("BBCBreaking", "Breaking news, sport, TV, radio and a whole lot more from the BBC.", 47_000_000, 3, 60_000, 8_500),
    ("BBCNews", "News, features and analysis from the BBC.", 14_000_000, 100, 700_000, 6_000),
    ("BBCWorld", "News, features and analysis from BBC World.", 41_000_000, 60, 580_000, 6_000),
    ("CNN", "It's our job to #GoThere and tell the most difficult stories. Comments aren't necessarily endorsements.", 63_000_000, 1_100, 426_000, 6_200),
    ("Reuters", "Top and breaking news, pictures and videos from Reuters.", 25_000_000, 1_400, 740_000, 6_400),
    ("AP", "News from The Associated Press, and a taste of the great journalism produced by AP members and customers.", 17_000_000, 7_000, 320_000, 5_900),
    ("NPR", "Public radio news, in your podcasts, on your radio, on your phone.", 9_000_000, 6_500, 280_000, 6_400),
    ("nytimes", "News tips? Share them here.", 55_000_000, 900, 410_000, 6_500),
    ("washingtonpost", "Democracy Dies in Darkness.", 19_000_000, 1_700, 360_000, 6_100),
    ("Guardian", "The Guardian. Independent journalism since 1821.", 11_000_000, 100, 460_000, 6_000),
    ("AJEnglish", "Breaking news from Al Jazeera English.", 9_600_000, 300, 424_800, 6_300),
    ("FinancialTimes", "World business, finance, and political news from the Financial Times.", 7_000_000, 1_200, 500_000, 6_000),
    ("business", "Bloomberg Business news.", 9_500_000, 800, 700_000, 6_000),
    ("TheEconomist", "News and analysis with a global perspective.", 28_000_000, 200, 200_000, 6_400),
    ("WIRED", "WIRED is where tomorrow is realised.", 11_000_000, 200, 350_000, 6_300),

    ("Apple", "", 10_500_000, 9, 6, 1_500),
    ("Microsoft", "We're empowering people and organizations to do more.", 14_000_000, 6_000, 25_000, 6_000),
    ("Google", "Like Google, but on Twitter.", 33_000_000, 100, 22_000, 6_000),
    ("Meta", "From the team at Meta.", 4_500_000, 400, 7_000, 5_500),
    ("amazon", "Hi, we're Amazon.", 6_500_000, 300, 36_000, 5_700),
    ("Tesla", "", 28_000_000, 0, 7_000, 4_000),
    ("netflix", "yes we are open. comments off until further notice.", 22_000_000, 700, 50_000, 5_500),
    ("Spotify", "Spotify is the official audio streaming brand of awkward dancing.", 4_500_000, 500, 70_000, 5_500),
    ("Adobe", "Creativity for all. Helping people create, interact, and inspire through digital experiences.", 1_400_000, 1_000, 30_000, 6_200),
    ("OpenAI", "AI research and deployment company.", 4_000_000, 50, 1_500, 2_500),

    ("Starbucks", "Inspiring and nurturing the human spirit. One person, one cup, one neighborhood at a time.", 11_000_000, 100_000, 95_000, 6_500),
    ("Nike", "If you have a body, you are an athlete.", 10_500_000, 200, 35_000, 6_000),
    ("McDonalds", "lovin beats hatin", 4_700_000, 200, 60_000, 5_900),
    ("CocaCola", "Refresh the World. Make a Difference.", 3_500_000, 70_000, 250_000, 6_400),
    ("Walmart", "Save Money. Live Better.", 1_300_000, 35_000, 90_000, 6_300),
    ("Target", "Hi, we're Target.", 2_000_000, 6_000, 130_000, 6_400),
    ("IKEAUSA", "Make every day brighter for the many people. Customer support: @IKEAUSAHelp", 600_000, 700, 160_000, 6_100),
    ("adidas", "Through sport, we have the power to change lives.", 4_300_000, 600, 80_000, 6_000),
    ("pepsi", "Welcome to the official feed of the unmistakably refreshing Pepsi.", 3_300_000, 1_300, 65_000, 5_800),
    ("Disney", "The official Disney Twitter feed.", 8_500_000, 500, 35_000, 6_000),

    ("espn", "ESPN on Twitter.", 45_000_000, 1_500, 700_000, 6_000),
    ("NBA", "The official account of the NBA.", 47_000_000, 2_000, 800_000, 6_300),
    ("NFL", "The official source for NFL news and broadcast.", 30_000_000, 8_000, 600_000, 6_400),
    ("premierleague", "The official Premier League account.", 35_000_000, 200, 350_000, 5_800),
    ("FIFAcom", "Official Twitter account of FIFA.", 17_000_000, 200, 230_000, 6_400),
    ("MLB", "The Official Twitter Account of MLB.", 11_000_000, 400, 650_000, 6_400),
    ("F1", "The Official Account of Formula 1, F1 World Championship.", 11_000_000, 100, 320_000, 5_400),
    ("ChampionsLeague", "The official UEFA Champions League account.", 30_000_000, 200, 250_000, 5_800),

    ("NASA", "There is space for everybody.", 75_000_000, 200, 78_000, 6_400),
    ("WHO", "We are the United Nations health agency. HealthForAll.", 12_000_000, 1_500, 90_000, 5_900),
    ("UN", "Official account of the United Nations. For peace, dignity and equality on a healthy planet.", 16_000_000, 5_000, 100_000, 6_400),
    ("UNICEF", "Working for children's rights, their survival, development and protection.", 10_500_000, 4_000, 130_000, 6_400),
    ("RedCross", "Helping people in emergencies in the U.S. and around the world.", 3_900_000, 8_000, 160_000, 6_400),
    ("NHSEngland", "Welcome to NHS England. Latest news, advice, blogs, podcasts and more.", 1_400_000, 4_000, 80_000, 5_500),
    ("NWS", "Official Twitter for the National Weather Service. Weather forecasts, watches and warnings.", 1_900_000, 200, 250_000, 6_300),
]


def evaluate():
    import numpy as np
    from api.app import predict, load_model, extract_numeric
    load_model()
    from api.app import _model, _numeric_mean, _numeric_std, _threshold
    rows = []
    fired = 0
    correct_with = 0
    correct_without = 0
    for username, bio, followers, following, tweets, age_days in orgs:
        profile = {
            "username": username,
            "bio": bio,
            "followers_count": followers,
            "following_count": following,
            "tweet_count": tweets,
            "favourites_count": max(int(tweets * 0.4), 50),
            "account_age_days": age_days,
            "is_verified": True,
            "has_default_avatar": False,
        }
        result_with = predict(profile)
        if result_with["override_applied"]:
            fired += 1
        raw = extract_numeric(profile)
        arr = np.array(raw, dtype=np.float32)
        if _numeric_mean is not None:
            arr = (arr - _numeric_mean) / _numeric_std
        prob_without = float(_model.predict_proba(arr.reshape(1, -1))[0, 1])
        delta = prob_without - _threshold
        if abs(delta) <= 0.1:
            label_without = "uncertain"
        elif delta > 0:
            label_without = "bot"
        else:
            label_without = "human"

        if result_with["label"] == "human":
            correct_with += 1
        if label_without == "human":
            correct_without += 1
        rows.append({
            "username": username,
            "label_with_override": result_with["label"],
            "score_with_override": result_with["bot_score"],
            "label_without_override": label_without,
            "score_without_override": int(round(prob_without * 100)),
            "override_fired": result_with["override_applied"] is not None,
        })

    summary = {
        "n_organisations": len(orgs),
        "override_fired_count": fired,
        "correctly_classified_human_with_override": correct_with,
        "correctly_classified_human_without_override": correct_without,
        "improvement": correct_with - correct_without,
        "rows": rows,
    }

    out_path = os.path.join(checkpoints, "orgs_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\norgs evaluated: {len(orgs)}")
    print(f"override fired on: {fired}/{len(orgs)} accounts")
    print(f"correctly classified as human without override: {correct_without}/{len(orgs)}")
    print(f"correctly classified as human with override: {correct_with}/{len(orgs)}")
    print(f"improvement from override: {correct_with - correct_without}")

    misses_with = [r for r in rows if r["label_with_override"] != "human"]
    if misses_with:
        print(f"\nstill misclassified with override ({len(misses_with)}):")
        for r in misses_with:
            print(f"  @{r['username']:<18} {r['label_with_override']:<9} ({r['score_with_override']}/100)")

    misses_without = [r for r in rows if r["label_without_override"] != "human"]
    if misses_without:
        print(f"\nmisclassified without override ({len(misses_without)}):")
        for r in misses_without:
            rescued = " (rescued by override)" if r["override_fired"] else ""
            print(f"  @{r['username']:<18} {r['label_without_override']:<9} ({r['score_without_override']}/100){rescued}")

    print(f"\nsaved to {out_path}")

if __name__ == "__main__":
    evaluate()