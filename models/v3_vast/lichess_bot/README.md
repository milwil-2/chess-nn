# Lichess BOT deployment (Phase E3)

This directory contains the configuration needed to deploy our v3_vast UCI
engine as a Lichess BOT account. It is intentionally **not active by default**
— the user must perform several manual steps first.

## Why is this gated behind manual approval?

Lichess enforces a **no-sandbagging rule**: a BOT account cannot start weak
and ramp up. Putting an underdeveloped engine on Lichess will lock the
account at an embarrassingly low rating that cannot be recovered. We should
therefore confirm our internal gauntlet Elo first (see `run_gauntlet.py`) and
only deploy once we're satisfied with v3_vast's strength.

Additionally, the BOT title upgrade is **one-way** — once granted, the
account can never play in non-bot pools again on that login.

## Step 1 — create the Lichess account

Visit <https://lichess.org/signup> and create an account. Suggested username:
`chess-nn-bot` (verify availability — issue #22 marked this as an open
decision).

Once signed up, generate a personal API token at
<https://lichess.org/account/oauth/token>. Required scopes:

- `bot:play`
- `challenge:read`
- `challenge:write`

Save the token — you'll paste it into `config.yml` below.

## Step 2 — email Lichess for the BOT upgrade

Send the following email to `contact@lichess.org`. **Do this only when you're
ready** — the upgrade is permanent.

```
Subject: BOT title upgrade request

Hi Lichess team,

Could you please upgrade the account "<username>" to BOT status?

The account will be driven by lichess-bot
(https://github.com/lichess-bot-devs/lichess-bot) bridging to our
in-house UCI engine (an AlphaZero-style neural network trained on
Lichess monthly PGNs, with MCTS at inference). Source:
https://github.com/milwil-2/chess-nn

I understand the upgrade is one-way and that the account cannot have
played any human games before the upgrade. I confirm no rated games
have been played from this account.

Thanks,
<your name>
```

Approval typically takes 24–48 hours. The account is restricted from
playing any rated games while waiting.

## Step 3 — install lichess-bot

```bash
git clone https://github.com/lichess-bot-devs/lichess-bot.git ~/lichess-bot
cd ~/lichess-bot
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 4 — configure

Copy `config.yml.template` from this directory into `~/lichess-bot/config.yml`
and edit:

1. Paste your API token into the `token:` field.
2. Confirm `engine.dir` and `engine.name` point to our `run.py` (already set
   in the template).
3. Confirm the auto-accept rules under `challenge:` match what you want.
   Defaults: rated only, time control between 5+3 and 15+10, opponent rating
   1500–2500.

## Step 5 — run

```bash
cd ~/lichess-bot
source .venv/bin/activate
python lichess-bot.py -v
```

Let it accumulate at least 30 rated games (a day or two of uptime depending
on challenge frequency). Then log the final Lichess rating in
`models/v3_vast/logs/elo_history.csv` alongside the gauntlet Elo for
cross-calibration.

## What this directory does NOT do

- It does **not** create the Lichess account.
- It does **not** send the email — copy/paste the template above yourself.
- It does **not** install lichess-bot.
- It does **not** include your API token.

All four steps are manual on purpose, per the no-sandbagging risk above.
