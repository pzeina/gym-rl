"""CONTACT pricing must give precision an actual defence (v1.11).

A CONTACT is adjudicated in three tiers by ``CohortEnv._report_contact``: new
intel pays ``contact_new``, a re-report whose intel has aged past
``contact_refresh_age`` is a legitimate refresh worth exactly 0, and anything
else is noise drawing ``contact_redundant``. ``transmission_cost`` rides on all
three.

At ``contact_redundant = -0.02`` the arithmetic made spam correct: a redundant
report cost -0.03 all-in against +0.49 for an informative one, so a policy
profited by reporting whenever precision stayed above **5.8%**.
fireteam_defend_v8 measured 0.38 (289 informative, 480 noise, N=100) — rational
play against a price with no teeth, not a training failure.

The regression hazard runs both ways, so both are pinned here: spam must be
-EV at plausible precision, AND an informative report must stay clearly worth
sending, because B5 showed that over-pricing a speech act suppresses the honest
one too. A cohort that stops reporting contacts blinds the whole picture.
"""

from cohort import make_env
from cohort.core.orders import MessageKind
from cohort.env.actions import CATALOG
from cohort.env.rewards import RewardConfig

STAY = 0
CONTACT = next(s.index for s in CATALOG if s.kind == "contact")


def _breakeven(cfg: RewardConfig) -> float:
    """Precision below which spamming stops paying."""
    gain = cfg.contact_new + cfg.transmission_cost
    cost = -(cfg.contact_redundant + cfg.transmission_cost)
    return cost / (gain + cost)


def test_spam_breakeven_is_a_real_constraint():
    """The defect: break-even at 0.058 meant no constraint at all."""
    cfg = RewardConfig()
    assert 0.25 <= _breakeven(cfg) <= 0.55, (
        f"break-even precision {_breakeven(cfg):.3f} is outside the intended band; "
        "too low is the v8 spam exploit, too high suppresses honest reporting"
    )


def test_informative_report_stays_clearly_worth_sending():
    """The B5 hazard: the honest act must keep a wide margin."""
    cfg = RewardConfig()
    informative = cfg.contact_new + cfg.transmission_cost
    assert informative > 0.4
    assert informative > -(cfg.contact_redundant + cfg.transmission_cost), (
        "an informative report must outweigh a redundant one, or reporting dies"
    )


def test_refresh_is_free_not_penalised():
    """A genuine refresh sits between the two and must not draw the penalty."""
    cfg = RewardConfig()
    assert cfg.contact_refresh_age > 0
    # the refresh tier adds nothing beyond airtime
    assert cfg.transmission_cost < 0
    assert cfg.contact_redundant < cfg.transmission_cost


def _env_with_a_visible_enemy(seed=12):
    """Flat ground with one enemy parked beside a soldier, so LOS is certain.

    fireteam_defend holds OpFor through the preparation period and the stock
    terrain blocks sight lines, so waiting for a contact to happen is flaky.
    The pricing under test does not care how the enemy got there.
    """
    env = make_env("fireteam_defend")
    env.reset(seed=seed)
    env.world.grid[:] = 0
    sender = env.roster.living[0]
    enemy = env.enemies[0]
    enemy.alive = True
    enemy.pos = (sender.pos[0] + 1, sender.pos[1])
    enemy.home = enemy.pos
    for other in env.enemies[1:]:
        other.alive = False
    assert env._visible_enemies(sender), "fixture failed to create a visible enemy"
    return env, sender


def _send_contact(env, sender):
    acts = {a: STAY for a in env.agents}
    acts[sender.callsign] = CONTACT
    _, rewards, _, _, _ = env.step(acts)
    return rewards[sender.callsign]


def test_duplicate_storm_is_penalised_more_than_airtime():
    """End-to-end: re-sending the same tick's intel costs real reward."""
    env, sender = _env_with_a_visible_enemy()
    first = _send_contact(env, sender)

    env.enemies[0].pos = (sender.pos[0] + 1, sender.pos[1])  # keep it in sight
    assert env._visible_enemies(sender)
    second = _send_contact(env, sender)

    assert second < first, "an immediate duplicate must be worth less than the original"
    assert second < RewardConfig().transmission_cost, (
        "a duplicate storm must cost more than plain airtime"
    )


def test_contact_still_reaches_the_transcript():
    """Pricing changes must not silence the message itself."""
    env, sender = _env_with_a_visible_enemy()
    before = len(env.transcript.messages)
    _send_contact(env, sender)
    kinds = [m.kind for m in env.transcript.messages[before:]]
    assert MessageKind.CONTACT in kinds
