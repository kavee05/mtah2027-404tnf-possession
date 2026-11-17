# Who Has the Ball: Technical Document

**Team 404_TNF** · More than a Hack 2027 · Phase 1 Ball Possession

## 1. The idea in short

We treat possession as a story that unfolds over time and not as a guess made one frame at a time. Every frame gives us some evidence about three possible states. The home team has the ball. The away team has the ball. Or the ball is dead. We then let a decoder find the single sequence of states that explains all that evidence best while paying a price every time the state changes.

That price is the important part. A real match has a few hundred possessions and not tens of thousands. Frame level methods flicker every time a defender gets close to the ball and each flicker creates a fake possession. The ranking metric counts whole possessions so those fake ones hurt a lot. Our decoder only switches when the evidence for a switch is strong enough to pay for it. When it does switch it puts the change at the frame where the evidence starts and not half a window later as a smoothing filter would.

## 2. Assets we used and assets we dropped

We used three files from the tactical recording.

The tracking metadata tells us which track slot belongs to which team. We read this mapping from the format specification in the file itself. We do not assume which slot numbers belong to which team. In this match the first 24 slots are home and the last 22 are away, but reading the mapping from the file means the code stays correct if another export orders them differently.

The tracking itself gives player and ball positions for every frame. We convert everything to metres using the 105.3 by 68 metre pitch from games.py.

The Smart Tagging gives tactical phases with a team tag and set pieces with a restart type.

We did not use the videos or the homography. The tracking already lives on the video timeline so the homography adds nothing for this question. Running our own detector on the video could recover some of the missing ball positions. We judged that it would not fit in the time we had. We also left out the broadcast recording because it is not synchronised with the tactical one.

## 3. How the pipeline works

### 3.1 Who is close to the ball

For every frame where the ball is visible we measure the distance from the ball to the nearest home player and to the nearest away player. Each distance becomes a control score that is close to one when the ball is at a player's feet and fades to zero beyond about two metres. The difference between the two scores is our evidence for home against away.

Two details matter here. When the ball moves faster than 8 metres per second it is usually a pass or a shot passing by someone, so we trust proximity less. When both teams are close or nobody is close the evidence is near zero. That means the possession simply stays with the last team, which is exactly what the challenge rules ask for during loose balls and duels.

### 3.2 A team model that learns from the tracking alone

The ball is missing on about 41 percent of graded frames and it tends to disappear at the busy moments. To say something useful during those gaps we train a small logistic regression that predicts the team in possession from the shape of both teams.

Its inputs are the difference in average position along the pitch, the difference in width and depth of each team, the difference in running speed, how far up the pitch the whole game is tilted, and one signed indicator per Smart Tagging phase. The model learns by itself whether a tag such as Build Up names the attacking team or the defending one.

The labels for training come from our own algorithm and not from people. We take frames where one team has a player within one metre of a slow ball and the other team has nobody within three metres. That gives about 17,500 confident frames. The model is trained separately for each half because the teams swap ends. On held out blocks of time it agrees with these pseudo labels 83 percent of the time in the first half and 85 percent in the second.

Its output is added to the evidence with a small weight of one twenty fifth per frame. Neighbouring frames are almost copies of each other, so one second of frames counts as roughly one real observation.

### 3.3 Spotting a dead ball

The dead ball state draws on four signals.

1. **Ball outside the lines.** If the tracked ball is more than one metre beyond a touchline or goal line we count that as strong evidence. Between 0.3 and 1 metre it counts as weak evidence.
2. **Players slowing down.** During stoppages players walk. We fit a two component Gaussian mixture to the log of the average player speed over the whole match. This is unsupervised. The slow component becomes evidence for a dead ball. We trust it less while the ball is visibly moving, so slow build up play is not mistaken for a stoppage.
3. **A moving ball at someone's feet.** This is counted as evidence that play is live.
4. **Set pieces from Smart Tagging.** Each set piece window wraps around a restart. Inside the window we look for the kick, which is the first moment the ball goes from standing still for at least ten frames to moving faster than 4 metres per second. Before the kick we add evidence for a dead ball. For two seconds after it we add evidence for live play. For about 1.6 seconds we add evidence for the restarting team. That team comes from the tag. Throw ins have no team tag, so there we use whichever team is closest to the ball at the kick. We found the kick from ball motion in 45 of the 59 set pieces and used the middle of the window for the rest.

### 3.4 Putting it together

All evidence is summed into a score per frame for each of the three states. A Viterbi decoder then finds the best path through each half. A change between home and away costs 8 nats. Going from live play to dead ball costs 8 nats. Restarting from a dead ball costs 4, because a restart is the normal way out of a stoppage. Frames before each kickoff are set to dead ball and frames outside the two graded halves are written as X.

## 4. Possession rules we implemented

**Controlled possession.** A team gains the ball only after its players have been clearly closer to it for long enough to outweigh the switching cost. As a rough guide that is about half a second of clean control, or longer when the ball is only partly seen. A single touch or a deflection does not pay for the switch so it does not count.

**Loose balls and aerial duels.** Evidence near zero means no change. The last team keeps the ball.

**Restarts.** The team that takes the set piece starts with the ball. Throw ins fall back to proximity at the moment of the kick.

**Dead ball.** We never assign a dead ball to a team. Dead ball is always D as the rules require.

## 5. How we set the numbers

Every constant was chosen from football and tracking common sense before looking at any labels. Examples are a control radius of about one metre, a fast ball at 8 metres per second and one effective observation per second. None of them were fitted to the public five minutes. Those five minutes were used only once at the end to confirm that frames line up and that our labels mean what we think.

On that window our file scores spellF1 0.507 and macroF1 0.718. The strongest official baseline scores 0.424 and 0.581. Team accuracy on live frames is 90 percent. We know that 17 possessions are far too few to rank anything, so we read this only as proof that nothing is broken.

Over the full match the decoder produces 254 possessions with a mean length of 15 seconds. That is in the range a real match has.

## 6. Training data

We used no human labelled data at all. Not from this game and not from any other game. The only labels in the pipeline are the proximity based pseudo labels described in section 3.2 and those are produced by code.

## 7. Limitations and what we would do next

We think we predict too much dead ball. The motion signal also fires during slow spells of play, and about 32 percent of all frames end up as D. A detector trained on whistle and restart patterns from other matches would fix most of this.

When the ball is missing for several seconds and possession changes during the gap, we only notice the change when the ball comes back. A ball detector on the tactical video would shorten those gaps. Interpolating the ball between sightings would also help.

Fouls and offsides without a tagged set piece are found only through the motion signal. That signal is slow to react.

The team model is linear and quite small. A sequence model trained with the same pseudo labels could read team shape much better.

## 8. Reproduction

```
pip install -r requirements.txt
python -m possession.pipeline --out submissions/404_TNF.csv
python evaluation/validate_submission.py submissions/404_TNF.csv
```

Put the data bundle in `data/` or set `MTAH_DATA_DIR`. The run takes about 15 seconds on a laptop and uses only numpy and pandas. It has no randomness, so it reproduces the submitted file byte for byte. Its MD5 is `75df2fe7390e41ec1dac25eb1f99fa3a`.
