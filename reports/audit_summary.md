# Recommendation System Audit Report

## Performance Summary
- **Baseline (SVD) RMSE**: 0.8846
- **Advanced (PMF) RMSE**: 0.8402
- **Relative Improvement**: 5.02%

## Conclusion
The PMF model is the production candidate.
## Interpretability Analysis
### Global Trends (Factor Analysis)
```
--- Global Latent Factor Trends ---

Factor 0 (Representative Movies):
  1. Hate (Haine, La) (1995)
  2. Happy Gilmore (1996)
  3. Alaska (1996)
  4. Cutthroat Island (1995)
  5. Godfather, The (1972)

Factor 1 (Representative Movies):
  1. Hugo Pool (1997)
  2. Kim (1950)
  3. Star Wars: Episode V - The Empire Strikes Back (1980)
  4. Amityville II: The Possession (1982)
  5. Winnie the Pooh and the Blustery Day (1968)

Factor 2 (Representative Movies):
  1. Spitfire Grill, The (1996)
  2. Day the Sun Turned Cold, The (Tianguo niezi) (1994)
  3. Heathers (1989)
  4. Rude (1995)
  5. Vegas Vacation (1997)
```
### Local Example
```
--- Local Interpretability ---
User 5300 -> Movie: To Wong Foo, Thanks for Everything! Julie Newmar (1995)
Strongest Driver: Latent Factor 24
Factor Contribution: -0.1005
```
