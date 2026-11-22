"""Budget-constrained telemetry-investigation environment and DQN agent.

State (25 values)
    * 12 log-scaled LSTM-AE channel reconstruction errors   log1p(error)
    * 12 investigation-status values  0 = not investigated, 0.5 = investigated & not associated,
                                      1 = investigated & associated (outcome of that investigation)
    * remaining budget / budget
Ground-truth annotations are never part of the initial observation; a channel's association status only
becomes visible after the agent has spent budget investigating it.
Action: one of 12 channels (already-investigated channels are masked out).
Reward: +1 if the investigated channel is annotated for the event, -1 otherwise.
Episode: one anomaly event, exactly `budget` investigations.
"""
import os
import numpy as np

N_CH = 12


class InvestigationEnv:
    def __init__(self, scores, targets, budget=5):
        self.scores = np.asarray(scores, dtype=np.float32)          # (n_cases, 12) log-scaled errors
        self.targets = np.asarray(targets, dtype=bool)              # (n_cases, 12) annotated channels
        self.budget = budget
        self.n_cases = len(self.scores)

    def reset(self, case_index):
        self.case = int(case_index)
        self.status = np.zeros(N_CH, dtype=np.float32)
        self.remaining = self.budget
        return self.obs()

    def obs(self):
        return np.concatenate([self.scores[self.case], self.status / 2.0,
                               [self.remaining / self.budget]]).astype(np.float32)

    def available(self):
        return self.status == 0

    def step(self, action):
        action = int(action)
        if self.status[action] != 0:
            raise ValueError("Channel already investigated (should be masked)")
        hit = bool(self.targets[self.case, action])
        self.status[action] = 2.0 if hit else 1.0
        self.remaining -= 1
        reward = 1.0 if hit else -1.0
        return self.obs(), reward, self.remaining <= 0, hit


def episode_metrics(selected, target_row):
    hits = int(target_row[selected].sum())
    steps = len(selected)
    return {"Hits": hits, "Steps": steps, "TotalReward": float(2 * hits - steps),
            "Precision_at_5": hits / steps, "Recall_at_5": hits / int(target_row.sum()),
            "UnnecessaryInvestigations": steps - hits}


def build_q_network(tf, state_size=2 * N_CH + 1, hidden=64):
    return tf.keras.Sequential([tf.keras.Input(shape=(state_size,)),
                                tf.keras.layers.Dense(hidden, activation="relu"),
                                tf.keras.layers.Dense(hidden, activation="relu"),
                                tf.keras.layers.Dense(N_CH)])


def greedy_rollout(q_fn, env, case_index):
    s = env.reset(case_index)
    chosen = []
    for _ in range(env.budget):
        q = np.array(q_fn(s[None, :]))[0]
        q[~env.available()] = -np.inf
        a = int(np.argmax(q))
        s, _, done, _ = env.step(a)
        chosen.append(a)
        if done:
            break
    return chosen


def evaluate(q_fn, scores, targets, budget=5):
    env = InvestigationEnv(scores, targets, budget)
    rows = []
    for i in range(len(scores)):
        sel = greedy_rollout(q_fn, env, i)           # the SAME case i is loaded and scored
        m = episode_metrics(np.array(sel), targets[i]); m["Selected"] = sel
        rows.append(m)
    return rows


def train_seed(cfg):
    """Train one DQN seed. Runs in its own process (one CPU thread) so several seeds run in parallel."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import random
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    rng = np.random.RandomState(seed)

    env = InvestigationEnv(cfg["train_scores"], cfg["train_targets"], cfg["budget"])
    q = build_q_network(tf, hidden=cfg["hidden"]); qt = build_q_network(tf, hidden=cfg["hidden"])
    qt.set_weights(q.get_weights())
    opt = tf.keras.optimizers.Adam(cfg["lr"])
    gamma = cfg["gamma"]

    @tf.function
    def q_values(x):
        return q(x, training=False)

    @tf.function
    def train_step(s, a, r, s2, done, mask2):
        q_next = qt(s2, training=False)
        q_next = tf.where(mask2, q_next, tf.fill(tf.shape(q_next), -1e9))      # masked max over available
        target = r + gamma * tf.reduce_max(q_next, axis=1) * (1.0 - done)
        with tf.GradientTape() as tape:
            qs = q(s, training=True)
            q_sa = tf.reduce_sum(qs * tf.one_hot(a, N_CH), axis=1)
            loss = tf.reduce_mean(tf.square(target - q_sa))
        opt.apply_gradients(zip(tape.gradient(loss, q.trainable_variables), q.trainable_variables))
        return loss

    cap = cfg["buffer"]; B = cfg["batch"]
    buf_s = np.zeros((cap, 2 * N_CH + 1), np.float32); buf_s2 = np.zeros_like(buf_s)
    buf_a = np.zeros(cap, np.int32); buf_r = np.zeros(cap, np.float32); buf_d = np.zeros(cap, np.float32)
    buf_m2 = np.zeros((cap, N_CH), bool); ptr = 0; size = 0

    eps = cfg["eps_start"]
    history, checkpoints = [], []
    best = (-np.inf, None, 0)
    for ep in range(1, cfg["episodes"] + 1):
        s = env.reset(rng.randint(env.n_cases)); total = 0.0; losses = []
        while True:
            avail = np.where(env.available())[0]
            if rng.rand() < eps:
                a = int(rng.choice(avail))
            else:
                qv = q_values(s[None, :]).numpy()[0]; qv[~env.available()] = -np.inf; a = int(np.argmax(qv))
            s2, r, done, _ = env.step(a)
            buf_s[ptr], buf_a[ptr], buf_r[ptr], buf_s2[ptr], buf_d[ptr], buf_m2[ptr] = s, a, r, s2, float(done), env.available()
            ptr = (ptr + 1) % cap; size = min(size + 1, cap)
            if size >= B:
                idx = rng.randint(size, size=B)
                losses.append(float(train_step(buf_s[idx], buf_a[idx], buf_r[idx], buf_s2[idx], buf_d[idx], buf_m2[idx])))
            s = s2; total += r
            if done:
                break
        eps = max(cfg["eps_min"], eps * cfg["eps_decay"])
        if ep % cfg["target_update"] == 0:
            qt.set_weights(q.get_weights())
        history.append({"Episode": ep, "Reward": total, "Epsilon": eps, "Loss": float(np.mean(losses)) if losses else np.nan})
        if ep % cfg["eval_every"] == 0:
            rows = evaluate(lambda x: q_values(x).numpy(), cfg["val_scores"], cfg["val_targets"], cfg["budget"])
            vr = float(np.mean([r_["TotalReward"] for r_ in rows]))
            checkpoints.append({"Episode": ep, "Validation_Reward": vr})
            if vr > best[0] + 1e-9:
                best = (vr, [w.copy() for w in q.get_weights()], ep)

    q.set_weights(best[1])
    weights_path = os.path.join(cfg["out_dir"], f"dqn_seed{seed}.keras")
    q.save(weights_path)
    return {"seed": seed, "history": history, "checkpoints": checkpoints, "best_episode": best[2],
            "best_val_reward": best[0], "weights": best[1], "model_path": weights_path}
