
## Initial thoughts
_Wed, Mar 25, 2026 at 11:11 PM_

The detect will possible be code to find the stale lines, outline process, potentionsl errors.
Detecting potential errors might be best with AI.
The analyze will probably be be code with a blend of Ai.
The brief will be AI. The challenge will be sending enough information to help AI understand the context.
I am thinking of showing a overview of bets that with icons for the issues. Possible difference of low, medium, high. I might have different agents for each type of discrepancy with unique system prompts. 

I am going to have the frontend as react and back-end as python.

At first I was thinking of having RAG and MCP for calculation. I might use MCP to allow the AI to get better context of the issue while keeping the context window small.

## Current plan
_Thu, Mar 26, 2026 at 08:07 AM_

I am going to calculate the odds for each line, then determine the best line. Then work on the best line. Finally, start on the AI integration. I am leaning towards using Claude with a previous integration I use. This emulates the terminal and allows me to use the Claude Scription without the Claude API penalty cost. I am thinking of testing different AI integrations to see if there are any differences. I will need to research how well different AI models handle sports betting. I have a feeling the frontier models will all perform similarly.

## Websockets
_Thu, Mar 26, 2026 at 09:29 AM_

I am going to use websockets for providing the latest status. This will help the website be responsive and display the latest without the webpage keep requesting for the latest data.

## Failover AI
_Thu, Mar 26, 2026 at 10:22 AM_

It might be a good idea if I support the ability to failover to another AI. AI is not reliable and Claude has been down frequently to high user traffic.

## Detect and Analyze Phase
_Thu, Mar 26, 2026 at 12:37 PM_

I want to have the following calculated during the detect and analyze phase. Most of this should be code and not benefit from AI, since it involves basic formulas. I am worried about performance and might have multiple processes running in parallel. I don't want to prematurely optimize until I review the performance.

Detech, Analyze & Brief

Core Odds Calculations
1. Implied Probability
•
Convert American odds to implied probabilities for any market (spread, moneyline, total)
•
e.g., DraftKings home moneyline of -228 → ~69.5% implied probability
2. Vig (Vigorish / Juice / Overround)
•
Calculate the bookmaker’s margin per market by summing implied probabilities of both sides
•
e.g., A spread at -111 / -111 has a ~3.9% vig
•
Compare vig across sportsbooks (the file notes Pinnacle has the tightest margins)
3. No-Vig / “Fair” Odds (True Probability)
•
Remove the vig to get the “true” implied odds for each market
•
Useful as a baseline to compare all books against
🔍 Cross-Sportsbook Analysis
4. Best Line Shopping
•
For each game & market, find which sportsbook offers the best odds for each side
•
e.g., Best home spread odds, best over odds, etc.
5. Arbitrage Detection
•
Check if combining the best odds from two different sportsbooks results in a guaranteed profit (combined implied probability < 100%)
6. Middles Detection
•
Find games where spread or total lines differ across books enough to bet both sides and potentially win both (e.g., one book has -4.5, another has +5.5)
7. Outlier / Anomaly Detection
•
The file explicitly says there are intentionally seeded anomalies to find:
•
Line outliers — spreads or totals that deviate significantly from the consensus
•
Odds outliers — moneyline prices far off from other books
•
Stale data — last_updated timestamps significantly older than others (e.g., PointsBet at 09:15 vs others at ~18:30)
📈 Statistical & Comparative
8. Consensus / Market Average
•
Calculate the average line and average odds across all 8 sportsbooks for each game & market
9. Standard Deviation of Lines/Odds
•
Measure how much disagreement there is between books on a given game
10. Expected Value (EV)
•
Using Pinnacle (sharp book) as the “true” line, calculate the EV of betting at each other sportsbook
•
Positive EV bets = potential value plays
11. Hold Percentage by Sportsbook
•
Rank sportsbooks by how much overall vig they charge across all games
⏱️ Data Freshness
12. Stale Line Detection
•
Flag records where last_updated is significantly older than the median for that game — these may be unreliable/stale odds

Power Rankings from Odds
•
Derive an implied team strength rating by aggregating each team’s moneyline across all their games
•
Build a relative power ranking of all 10 games’ teams purely from how the market prices them
2. Sportsbook “Sharpness” Score
•
Use Pinnacle (known sharp book) as the benchmark and score each sportsbook by how closely their lines track Pinnacle’s
•
A book that consistently deviates may be slower to update or cater more to recreational bettors
3. Correlation Between Markets
•
Check if the spread and moneyline imply the same win probability for each book
•
When they disagree (e.g., the spread implies 65% but the moneyline implies 60%), that’s a market inconsistency worth flagging
4. Synthetic Hold-Free Market
•
Build a “perfect book” by combining the best available odds across all 8 sportsbooks for every side of every market
•
Calculate the total edge a sharp bettor would have if they could always shop for the best line
5. Odds Movement Inference via Staleness
•
Since last_updated varies, compare stale odds vs. fresh odds for the same game
•
The direction of movement (e.g., line moved from -4.5 to -5.5) tells you where the sharp money went
6. Kelly Criterion Bet Sizing
•
Using no-vig “true” probabilities from Pinnacle, calculate the optimal Kelly bet size for every +EV opportunity at other books
•
Shows not just what to bet, but how much
7. Entropy-Based Market Efficiency
•
Calculate the Shannon entropy of implied probabilities across books for each game
•
Higher entropy = more disagreement between books = potentially more exploitable
8. Arbitrage Profit Curves
•
For each game, calculate the arb profit (if any) and plot how it changes as you mix different book combinations
•
Identify the most profitable book pairing combinations
9. Cluster Analysis on Sportsbooks
•
Group sportsbooks by how similarly they price lines (e.g., do DraftKings and FanDuel always agree? Does Pinnacle stand alone?)
•
Reveals which books likely share the same odds feed or risk model
10. Implied Total Margin of Victory
•
Combine the spread and total to estimate the implied final score for each team
•
e.g., spread -5.5 + total 220 → Home ~112.75, Away ~107.25
11. “Wisdom of Crowds” Consensus vs. Sharp Book
•
Average all 8 books’ implied probabilities (crowd wisdom) and compare to Pinnacle alone (sharp wisdom)
•
When they diverge significantly, it highlights where recreational books are mispricing
12. Cross-Market Arbitrage
•
Look for arb opportunities across market types — e.g., a moneyline at one book vs. an alternative spread at another that covers the same outcome at better combined value
🎯 Anomaly-Specific (The Data Has Seeded Anomalies!)
13. Z-Score Anomaly Detection
•
For each game’s market, compute the mean and standard deviation across all books, then flag any odds with a z-score > 2 as an anomaly
14. Timestamp Anomaly Scoring
•
Score each record’s last_updated relative to the game’s commence_time — the further out it is, the more suspect the line

Data Science & ML-Driven Approaches
1. Graph-Based Anomaly Detection (CNN on Odds Patterns)
•
Convert each game’s odds across all 8 sportsbooks into a visual graph/heatmap and use pattern recognition to spot abnormal shapes
•
Research from Nature/Scientific Reports shows this method achieves 92%+ accuracy in detecting fixed or manipulated matches
2. Unsupervised Clustering for Anomaly Flagging
•
Use K-Nearest Neighbor (KNN) or Random Forest classifiers on your 80 records to flag odds that deviate from normative patterns without needing labeled training data
•
Analyze features like: odds deviation from mean, timestamp staleness, line movement direction
3. GAMLSS Statistical Modeling
•
Apply a Generalized Additive Model for Location, Scale, and Shape to the odds distributions — this models not just the average but the variance and skewness of odds across books, making anomalies mathematically pop out
📐 Advanced Mathematical Techniques
4. Bayesian Updating of True Probabilities
•
Start with a prior probability (e.g., from Pinnacle’s no-vig line) and update it using each additional sportsbook’s odds as evidence
•
The posterior probability after incorporating all 8 books gives you a more robust “true” probability than any single book
5. Poisson Distribution for Score Prediction
•
Use the implied total and spread to derive expected goals/points per team, then apply a Poisson model to calculate the probability of every possible final score
•
This lets you price props, quarters, and alternative lines yourself
6. Closing Line Value (CLV) Simulation
•
Since you have last_updated timestamps, simulate which odds would be the “closing line” (latest update) and measure how much value the earlier lines offered
•
Consistently beating the closing line is the #1 indicator of a sharp bettor
7. Kelly Criterion with Fractional Sizing
•
Formula: (bp - q) / b where b = decimal odds - 1, p = your true probability, q = 1 - p
•
Apply fractional Kelly (e.g., half-Kelly) to every +EV opportunity to build a full bankroll management simulation across all 10 games
🧩 Cross-Market & Structural Analysis
8. Market Consistency Scoring (Spread vs. Moneyline vs. Total)
•
Each market implies a win probability — when the spread implies 68% but the moneyline implies 62% at the same book, that book has an internal inconsistency you can exploit
9. Bookmaker Margin Decomposition (Shin Model)
•
Instead of splitting vig equally, use the Shin method to allocate more vig to the longshot side (which is how books actually operate)
•
This gives more accurate “true” probabilities than naive vig removal
10. Synthetic “Perfect Book” Construction
•
Combine the best line from each sportsbook for every side of every market
•
Calculate the total hold of this synthetic book — if it’s negative, guaranteed arbitrage exists across those books
11. Implied Score Matrix
•
Combine spread + total to derive each team’s implied final score:
•
Home = (Total + Spread) / 2
•
Away = (Total - Spread) / 2
•
Compare these across all 8 books to see where scores diverge most
12. Odds Elasticity / Sensitivity Analysis
•
Calculate how much a 0.5-point line change impacts the odds across books
•
Books with higher elasticity (big odds swings for small line changes) may be less confident in their pricing
🕸️ Network & Relationship Analysis
13. Sportsbook Correlation Network
•
Build a correlation matrix of how closely each pair of sportsbooks tracks each other
•
Reveals which books share odds feeds (high correlation) vs. which are independently pricing (potential value sources)
14. Information Flow Analysis via Timestamps
•
Map the last_updated order across books to infer which sportsbook moves first (market leader) and which follow
•
The leader is typically the sharpest; followers that lag create exploitable windows

## Notes about the devnotes
_Thu, Mar 26, 2026 at 12:39 PM_

I have a generative process for the devnotes. This is because if I deploy to a server, all the Claude data will be lost.

## AI Summarizer
_Thu, Mar 26, 2026 at 06:55 PM_

I believe it will be beneficial to summarize the AI summary to a quick and simple overview of the analysis

## I started to add the AI code
_Thu, Mar 26, 2026 at 7:36 PM_

## Ai versus code
_Thu, Mar 26, 2026 at 7:49 PM_

I don't trust AI to do basic arithmetic including greater than or less that. I am planning to use MCP to return the best or worst in a json format. This way AI does not calculate anything. The Json will need to return a description that is hard coded. From this the AI can take the information and rationalize the reason and thoughts.

## Force ai to use the MCP
_Thu, Mar 26, 2026 at 7:53 PM_

I will need to add some guard rails, sub agent and/or system prompt to make sure the AI use the MCP. I am thinking of stressing testing this to make sure the mcp works.

## Historical betting might be helpful for this project
_Thu, Mar 26, 2026 at 8:18 PM_

## Long term improvement
_Fri, Mar 27, 2026 at 11:42 AM_

Long term the data files should be cleaned up to minimize the disk space usage.
