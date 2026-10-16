# Born to Cover 18

A spread reference board for **college football and the NFL**. For every game it shows:

- the model's **fair spread** and how far it is from the sportsbook line (the edge)
- the **opening line, current line and movement**, plus the best number available
- **DraftKings splits** (% of bets vs % of money), downloaded automatically, with bigger-money and reverse-line-move flags
- **lookahead / letdown / sandwich spots**, rest days and kickoff **weather**
- a **verdict for any side you like** ("Model backs it", "No real edge", "Model says stay off")
- a **pick log** that grades results and your **closing-line value** automatically

Nothing needs to be typed in. Set your stipulations once in **My rules** (sidebar sliders: picks per week, minimum edge, biggest spread you'll bet, how many signals must agree, only-when-the-opponent-is-in-a-spot, and more). The shortlist only shows games that pass them. To check one of your own plays, pick the game and side in *Check my pick* and press **Log this pick**.

## What changed from v17.1

v17.1 had 125 files of stacked patches. Its 72% records came from choosing the best of 120+ versions using the same seasons it was graded on. Its "60–65%" estimates were mostly totals, inflated by hand-typed point values. v18 is about 15 files. Every number in the app comes from a test where each season was predicted by a model that never saw that season.

## Honest performance (walk-forward 2018–2025)

| | vs closing line | vs opening line (CFB) |
|---|---|---|
| Every game | ~50% | ~50% |
| Top 2 disagreements per week, CFB | 52% | **58% (117-85), line moved +0.9 pts toward the model** |
| Top 2 disagreements per week, NFL | 54% | no opener history yet |

**The model's information gets priced in during the week.** Its best use is betting early, Sunday night and Monday, when it disagrees with the opener. At game-day lines, use it as a fair-number check and a veto. The NFL "model is against this side by 4.5+ points" spots covered only 45.7%.

## Setup (about 15 minutes)

1. **New GitHub repo.** Create `BornToCoverModel18` and upload every file and folder from this zip, including `models/`, `data/history/` and `.github/`.
2. **Streamlit.** Deploy a new app from that repo with main file `app.py`. In *Settings → Secrets*, paste the lines from `secrets.example.toml` with your real keys (the same CFBD, Odds API and R2 keys as before).
3. **Automatic line snapshots.** In GitHub, open *Settings → Secrets and variables → Actions* and add the same six keys as repository secrets. The workflow in `.github/workflows/snapshots.yml` then saves lines and splits 9 times a week (about 155 of your 500 monthly Odds API credits). This gives you true openers and closing lines even when the app isn't open. You can run it any time from the *Actions* tab with *Run workflow*.
4. Open the app. It updates stats, lines and splits by itself when they're stale. **Update now** forces an update.

## Weekly rhythm

- **Sunday night / Monday:** check *This week*. The top cards are the biggest disagreements with the openers, which is where the model has historically been worth the most.
- **Game day:** use *Check my pick* on your own plays. If the model says "stay off", get a better number or pass.
- **After the games:** *My picks* grades everything and shows average CLV. Positive CLV over 100+ bets is the clearest proof that your picks have an edge.

## Research behind it

- **Opponent-adjusted efficiency ratings**, the approach behind Massey-Peabody, which reported about 56% ATS over 500+ public picks.
- **Market-implied team ratings** taken from earlier closing lines.
- **Line timing.** Our model's top picks beat openers far more often than closing lines, and lines tend to overreact (Simon, *Management Science* 2024).
- **Tested and shown with their real records:** lookahead and letdown spots, power conference vs Group of 5, and home underdogs. None beat the closing line on their own. Reverse line movement and splits can't be tested yet because there's no free history of splits. The app saves every download, so your own pick log will answer it.

## Data sources and limits

- **NFL:** nflverse schedule, scores, closing lines and play-by-play. Free, no key.
- **CFB:** CollegeFootballData API, roughly 4 calls per update (free tier: 1,000 per month). Opening lines come from CFBD.
- **Lines:** The Odds API, 2 credits per league per update. One request covers DraftKings, FanDuel, BetMGM, Caesars, Fanatics, theScore (ESPN Bet), BetRivers, Pinnacle, LowVig and BetOnline.
- **Splits:** the DraftKings Network splits page, with VSiN's copy of the same DraftKings data as a fallback. Both sites' robots.txt allow it, and the app makes only a few requests per update. If DraftKings changes its page layout, the *DraftKings splits* tab has a hidden paste box as a backup.
- **Weather:** Open-Meteo, free.

## Retraining

```bash
python scripts/train.py          # rebuilds models/model_CFB.json and models/model_NFL.json
python -m unittest discover -s tests
```

Retrain in the off-season after adding the finished season to `data/history/`.
