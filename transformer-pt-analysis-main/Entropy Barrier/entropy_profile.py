#!/usr/bin/env python3
"""Fixed-log-training-loss entropy via continuous-parameter Wang--Landau.

See README_entropy_profile.md for the measure, finite-window limit and diagnostics.
This is a correctness-oriented random-walk baseline, not a fast MD sampler.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def stable_ce(logits, labels):
    """Per-example CE, including the '=' output class, without 1-epsilon rounding."""
    z = logits.double()
    margins = z - z.gather(1, labels[:, None])
    margins = margins.scatter(1, labels[:, None], -torch.inf)
    return F.softplus(torch.logsumexp(margins, dim=1))


def diagonal_mass(logits):
    """logits[a,b,c], c=0..p-1; class center, FFT over a,b, exclude DC."""
    centered = logits.double() - logits.double().mean(dim=-1, keepdim=True)
    power = torch.fft.fft2(centered, dim=(0, 1), norm="ortho").abs().square().sum(-1)
    # Explicit masking avoids subtracting nearly equal, large DC terms.
    power = power.clone()
    power[0, 0] = 0
    total = power.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("Diagonal mass is undefined: no finite non-DC power.")
    return power.diagonal().sum() / total


def reflected(values, bound):
    """Fold arbitrarily large jumps into [-bound,bound]; symmetric transition."""
    return bound - np.abs((values + bound) % (4 * bound) - 2 * bound)


def bin_of(y, edges):
    if not math.isfinite(y) or y < edges[0] or y > edges[-1]:
        return None
    return min(int(np.searchsorted(edges, y, side="right")) - 1, len(edges) - 2)


def flatness(hist):
    return float(hist.min() / hist.mean()) if hist.sum() else 0.0


def estimate_barrier(y, entropy, counts, left, right, min_visits=20):
    """Two specified phase bands; never bridge an unobserved interval."""
    empty = {"status": "insufficient_coverage", "delta_s_left": None, "delta_s_right": None}
    good = np.isfinite(entropy) & (counts >= min_visits)
    li = np.flatnonzero((y >= left[0]) & (y <= left[1]) & good)
    ri = np.flatnonzero((y >= right[0]) & (y <= right[1]) & good)
    if not len(li) or not len(ri):
        return empty
    i = int(li[np.argmax(entropy[li])])
    j = int(ri[np.argmax(entropy[ri])])
    if j <= i + 1 or not good[i:j + 1].all():
        return empty
    k = i + int(np.argmin(entropy[i:j + 1]))
    if k in (i, j) or entropy[k] >= min(entropy[i], entropy[j]):
        return {**empty, "status": "no_resolved_interior_trough"}
    return {"status": "candidate_projected_barrier", "y_left": float(y[i]),
            "y_trough": float(y[k]), "y_right": float(y[j]),
            "delta_s_left": float(entropy[i] - entropy[k]),
            "delta_s_right": float(entropy[j] - entropy[k])}


class Observables:
    def __init__(self, model, p, train_frac, data_seed, batch_size):
        from transformer import _make_pair_split
        self.model, self.p, self.batch_size = model, p, batch_size
        dev = next(model.parameters()).device
        ab = torch.cartesian_prod(torch.arange(p, device=dev), torch.arange(p, device=dev))
        self.tokens = torch.cat((ab, torch.full((p * p, 1), p, device=dev)), dim=1)
        self.labels = (ab[:, 0] + ab[:, 1]) % p
        pairs, _ = _make_pair_split(p, train_frac, data_seed)
        self.mask = torch.zeros(p * p, dtype=torch.bool, device=dev)
        for a, b in pairs:
            self.mask[a * p + b] = True
        if not self.mask.any():
            raise ValueError("Empty training split.")
        self.split_hash = hashlib.sha256(self.mask.cpu().numpy().tobytes()).hexdigest()

    @torch.no_grad()
    def table(self):
        self.model.eval()
        return torch.cat([self.model(t)[:, -1, :].double()
                          for t in self.tokens.split(self.batch_size)], dim=0)

    @torch.no_grad()
    def evaluate(self, theta):
        vector_to_parameters(theta, self.model.parameters())
        logits = self.table()
        loss = stable_ce(logits[self.mask], self.labels[self.mask]).mean().item()
        if not math.isfinite(loss) or loss <= 0:
            raise ValueError("Training loss is nonfinite or underflowed; try --dtype float64.")
        y = diagonal_mass(logits[:, :self.p].reshape(self.p, self.p, self.p)).item()
        return math.log(loss), y

    @torch.no_grad()
    def match_loss(self, theta, x_star):
        """Initialization only: positive output scaling preserves spectral mass."""
        vector_to_parameters(theta, self.model.parameters())
        logits = self.table()[self.mask]
        labels = self.labels[self.mask]
        target = math.exp(x_star)
        current = stable_ce(logits, labels).mean().item()
        if abs(math.log(current) - x_star) < 1e-5:
            return theta.clone(), 1.0
        margins = logits.gather(1, labels[:, None]) - logits
        margins.scatter_(1, labels[:, None], torch.inf)
        if (margins <= 0).any() or target >= math.log(logits.shape[1]):
            raise ValueError("Output scaling requires strictly correct predictions on all training "
                             "pairs and target loss < log(vocab). Choose a fitted checkpoint or "
                             "--no-match-loss if it already lies in the requested shell.")
        lo, hi = 0.0, 1.0
        while stable_ce(logits * hi, labels).mean().item() > target:
            hi *= 2
            if hi > 1e6:
                raise ValueError("Could not bracket the target loss by output scaling.")
        for _ in range(70):
            mid = (lo + hi) / 2
            if stable_ce(logits * mid, labels).mean().item() > target:
                lo = mid
            else:
                hi = mid
        scale = (lo + hi) / 2
        self.model.out_proj.weight.mul_(scale)
        self.model.out_proj.bias.mul_(scale)
        return parameters_to_vector(self.model.parameters()).detach().clone(), scale


def load_model(path, heads, device, dtype):
    from transformer import MinimalTransformer
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    vocab, width = state["token_embed.weight"].shape
    layers = len([k for k in state if k.endswith("in_proj_weight")])
    seq_len = state["pos_embed.weight"].shape[0]
    if seq_len < 3:
        raise ValueError("This runner expects the repository's [a,b,=] input format.")
    model = MinimalTransformer(vocab_size=vocab, d_model=width, n_heads=heads,
                               num_layers=layers, max_seq_len=seq_len,
                               hidden_mlp=state["mlp_layers.0.net.0.weight"].shape[0])
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=getattr(torch, dtype)).eval()
    return model, vocab - 1


def checkpoint_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def setup(args):
    model, p = load_model(args.mem_checkpoint, args.n_heads, args.device, args.dtype)
    obs = Observables(model, p, args.train_frac, args.data_seed, args.batch_size)
    mem = parameters_to_vector(model.parameters()).detach().clone()
    other, p_other = load_model(args.gen_checkpoint, args.n_heads, args.device, args.dtype)
    if p != p_other or [(k, v.shape) for k, v in model.state_dict().items()] != [
            (k, v.shape) for k, v in other.state_dict().items()]:
        raise ValueError("Checkpoint architectures differ.")
    gen = parameters_to_vector(other.parameters()).detach().clone()
    del other
    original = {"mem": dict(zip(("x", "y"), obs.evaluate(mem))),
                "gen": dict(zip(("x", "y"), obs.evaluate(gen)))}
    x_star = args.x_star if args.x_star is not None else original["mem"]["x"]
    if args.loss_star is not None:
        x_star = math.log(args.loss_star)
    if not -700 < x_star < 700:
        raise ValueError("Target log loss must be between -700 and 700 for stable float64 evaluation.")
    scales = {"mem": 1.0, "gen": 1.0}
    if args.command != "inspect" and not args.no_match_loss:
        mem, scales["mem"] = obs.match_loss(mem, x_star)
        gen, scales["gen"] = obs.match_loss(gen, x_star)
    bound = args.box_bound or args.box_factor * max(mem.abs().max().item(), gen.abs().max().item())
    metadata = {"p": p, "parameters": mem.numel(), "x_star": x_star,
                "loss_star": math.exp(x_star), "box_bound": bound,
                "original_checkpoints": original, "output_scaling": scales,
                "split_sha256": obs.split_hash,
                "checkpoint_sha256": {"mem": checkpoint_hash(args.mem_checkpoint),
                                      "gen": checkpoint_hash(args.gen_checkpoint)}}
    return obs, {"mem": mem, "gen": gen}, metadata


class ShellWangLandau:
    """Uniform Lebesgue measure in a box intersected with a log-loss/y window.

    State-independent block choice + reflected Gaussian proposals are symmetric.
    Hard-shell rejection followed by min(1, exp(V_old-V_new)) gives the frozen
    target 1_shell * exp(-V[bin(y)]). Update the *occupied* bin after every move.
    """
    def __init__(self, evaluate, theta, edges, x_star, half_width, bound,
                 step_size, block_size, seed, log_f=1.0):
        self.evaluate = evaluate
        self.theta = theta.clone()
        self.edges = np.asarray(edges)
        self.x_star, self.half_width, self.bound = x_star, half_width, bound
        self.step_size, self.block_size = step_size, min(block_size or theta.numel(), theta.numel())
        self.rng = np.random.default_rng(seed)
        self.x, self.y = evaluate(self.theta)
        self.current_bin = bin_of(self.y, self.edges)
        if self.theta.abs().max().item() > bound:
            raise ValueError("Starting checkpoint is outside the parameter box.")
        if abs(self.x - x_star) > half_width or self.current_bin is None:
            raise ValueError(f"Start (x={self.x:.8g}, y={self.y:.8g}) is outside the shell/y range.")
        n = len(edges) - 1
        self.bias = np.zeros(n)
        self.hist = np.zeros(n, dtype=np.int64)
        self.visits = np.zeros(n, dtype=np.int64)
        self.production = np.zeros(n, dtype=np.int64)
        self.prod_x_sum = np.zeros(n)
        self.log_f, self.stage, self.steps = log_f, 0, 0
        self.accepted = self.shell_rejects = self.y_rejects = self.invalid = 0
        self.round_trips = 0
        self.last_edge = None
        self.half_trips = 0
        self.adaptation_done = False
        self.history = []

    def move(self, adapt=True, collect=False):
        indices = self.rng.choice(self.theta.numel(), self.block_size, replace=False)
        idx = torch.as_tensor(indices, device=self.theta.device)
        proposal = self.theta.clone()
        # Fixed absolute noise scale: no state-dependent proposal-density bias.
        values = self.theta[idx].double().cpu().numpy()
        values = reflected(values + self.rng.normal(0, self.step_size, len(indices)), self.bound)
        proposal[idx] = torch.as_tensor(values, device=proposal.device, dtype=proposal.dtype)
        accepted = False
        try:
            x, y = self.evaluate(proposal)
        except ValueError:
            x, y = math.nan, math.nan
        b = bin_of(y, self.edges)
        if not math.isfinite(x) or not math.isfinite(y):
            self.invalid += 1
        elif abs(x - self.x_star) > self.half_width:
            self.shell_rejects += 1
        elif b is None:
            self.y_rejects += 1
        elif math.log(max(self.rng.random(), np.finfo(float).tiny)) < min(0.0, self.bias[self.current_bin] - self.bias[b]):
            self.theta, self.x, self.y, self.current_bin = proposal, x, y, b
            self.accepted += 1
            accepted = True
        b = self.current_bin
        if adapt:
            self.bias[b] += self.log_f
            self.hist[b] += 1
            self.visits[b] += 1
        if collect:
            self.production[b] += 1
            self.prod_x_sum[b] += self.x
        if b in (0, len(self.bias) - 1) and b != self.last_edge:
            if self.last_edge is not None:
                self.half_trips += 1
            self.last_edge = b
            self.round_trips = self.half_trips // 2
        self.steps += 1
        return accepted

    def refine(self, threshold, min_visits):
        if self.hist.min() >= min_visits and flatness(self.hist) >= threshold:
            self.history.append({"step": self.steps, "log_f": self.log_f,
                                 "flatness": flatness(self.hist), "min_visits": int(self.hist.min())})
            self.bias -= self.bias.max()
            self.hist[:] = 0
            self.log_f /= 2
            self.stage += 1
            return True
        return False

    def state_dict(self):
        # Plain data/tensors keep checkpoints compatible with weights_only=True.
        state = {k: v for k, v in vars(self).items() if k not in ("evaluate", "rng")}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                state[k] = torch.from_numpy(v.copy())
        state["theta"] = self.theta.cpu()
        state["rng_state_json"] = json.dumps(self.rng.bit_generator.state)
        return state

    def load_state(self, state):
        for k, v in state.items():
            if k == "rng_state_json":
                self.rng.bit_generator.state = json.loads(v)
            elif k == "theta":
                self.theta = v.to(self.theta.device)
            elif isinstance(v, torch.Tensor):
                setattr(self, k, v.cpu().numpy().copy())
            else:
                setattr(self, k, v)


def save_run(out, sampler, metadata, args, progress):
    state = {"sampler": sampler.state_dict(), "metadata": metadata,
             "arguments": vars(args), "progress": progress}
    tmp = out / "sampler.tmp"
    torch.save(state, tmp)
    tmp.replace(out / "sampler.pt")
    export_results(out, sampler, metadata, args, progress)


def export_results(out, s, metadata, args, progress):
    y = (s.edges[1:] + s.edges[:-1]) / 2
    density = np.full(len(y), np.nan)
    observed = s.production > 0
    # Frozen-bias histogram identity: g_j proportional to H_j exp(V_j).
    density[observed] = s.bias[observed] + np.log(s.production[observed]) - np.log(np.diff(s.edges)[observed])
    if observed.any():
        density[observed] -= density[observed].max()
    bias = s.bias - s.bias.max()
    bias[s.visits == 0] = np.nan
    mean_x = np.divide(s.prod_x_sum, s.production, out=np.full(len(y), np.nan), where=observed)
    with open(out / "profile.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "entropy_relative_nats", "wl_bias_relative_nats", "wl_visits", "production_visits", "mean_log_loss"])
        writer.writerows(zip(y, density, bias, s.visits, s.production, mean_x))
    barrier = estimate_barrier(y, density, s.production, args.left_phase, args.right_phase, args.min_visits)
    ready = s.adaptation_done and (s.production >= args.min_visits).all() and s.round_trips >= 2
    summary = {**metadata, "measure": "uniform parameters in [-B,B]^N; finite log-loss shell",
               "half_width": args.half_width, "y_range": [args.y_min, args.y_max],
               "status": "candidate_profile_requires_replica_and_window_checks" if ready else "incomplete_sampling",
               "wl_schedule_complete": s.adaptation_done, "log_f": s.log_f, "stages": s.stage,
               "steps": s.steps, "acceptance_rate": s.accepted / max(1, s.steps),
               "shell_rejections": s.shell_rejects, "y_rejections": s.y_rejects, "invalid_proposals": s.invalid,
               "visited_bins": int((s.visits > 0).sum()), "bins": len(y),
               "production_visited_bins": int(observed.sum()), "round_trips": s.round_trips,
               "current_stage_flatness": flatness(s.hist), "production_flatness": flatness(s.production),
               "stage_history": s.history, "progress": progress,
               "barrier_candidate": barrier,
               "caution": "Finite shell and explored component only. Flatness does not establish ergodicity. "
                          "A missing trough or missing coverage is not evidence of no barrier."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, constrained_layout=True)
    axes[0].plot(y, bias, "o--", ms=2, alpha=.6, label="adaptive WL bias (diagnostic)")
    axes[0].plot(y, density, "o-", ms=3, label="frozen-bias production estimate")
    if not observed.any():
        axes[0].text(.5, .5, "No production estimate yet", transform=axes[0].transAxes,
                     ha="center", va="center", color="tab:red")
    axes[0].set_ylabel("Relative entropy [nats]")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"x*={metadata['x_star']:.5g}, half-width={args.half_width:g}\n{summary['status']}")
    axes[1].stairs(s.visits, s.edges, label="WL visits")
    axes[1].stairs(s.production, s.edges, label="production visits")
    axes[1].set_ylabel("Visits")
    axes[1].legend()
    axes[2].plot(y, mean_x, "o-", ms=3)
    axes[2].axhline(metadata["x_star"], color="k", ls="--")
    axes[2].axhspan(metadata["x_star"] - args.half_width, metadata["x_star"] + args.half_width, alpha=.12)
    axes[2].set_ylabel("Mean log training loss")
    axes[2].set_xlabel("Diagonal spectral mass y")
    fig.savefig(out / "profile.png", dpi=160)
    plt.close(fig)


def run(args):
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("Output directory is not empty; use a new directory or --resume.")
    obs, seeds, metadata = setup(args)
    s = ShellWangLandau(obs.evaluate, seeds[args.start], np.linspace(args.y_min, args.y_max, args.bins + 1),
                       metadata["x_star"], args.half_width, metadata["box_bound"],
                       args.step_size, args.block_size, args.seed, args.log_f)
    progress = {"adapt_steps": 0, "burn_steps": 0, "production_steps": 0}
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        mutable = {"resume", "output", "steps", "production_steps", "burn_in", "log_every", "save_every", "batch_size", "device", "threads"}
        for k, v in vars(args).items():
            if k not in mutable and v != state["arguments"][k]:
                raise ValueError(f"Resume mismatch for --{k.replace('_', '-')}; use original setting.")
        if metadata != state["metadata"]:
            raise ValueError("Checkpoint contents, split or numerical observables changed since saved run.")
        s.load_state(state["sampler"])
        progress = state["progress"]
        if progress["production_steps"] and args.burn_in != state["arguments"]["burn_in"]:
            raise ValueError("Cannot change burn-in after production started.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    print("Training split reconstructed: verify --data-seed and --train-frac match training.", flush=True)
    started = time.monotonic()
    trace_path = out / "trace.csv"
    with open(trace_path, "a", newline="") as trace:
        writer = csv.writer(trace)
        if trace_path.stat().st_size == 0:
            writer.writerow(["step", "phase", "log_loss", "diagonal_mass", "accepted", "log_f", "stage_flatness"])
        try:
            for phase, limit in (("adapt", args.steps), ("burn", args.burn_in), ("production", args.production_steps)):
                if phase == "adapt" and s.adaptation_done:
                    continue
                while progress[phase + "_steps"] < limit:
                    accepted = s.move(adapt=phase == "adapt", collect=phase == "production")
                    progress[phase + "_steps"] += 1
                    done = False
                    if phase == "adapt" and s.steps % args.check_every == 0:
                        s.refine(args.flatness, args.min_visits)
                        if s.log_f <= args.log_f_min:
                            s.adaptation_done = done = True
                    if s.steps % args.log_every == 0:
                        writer.writerow([s.steps, phase, s.x, s.y, int(accepted), s.log_f, flatness(s.hist)])
                        trace.flush()
                        print(f"{phase:10} step={s.steps} x={s.x:.5g} y={s.y:.5f} "
                              f"accept={s.accepted/s.steps:.3f} bins={(s.visits>0).sum()}/{args.bins} "
                              f"ln(f)={s.log_f:.3g} round_trips={s.round_trips}", flush=True)
                    if s.steps % args.save_every == 0:
                        save_run(out, s, metadata, args, progress)
                    if done:
                        break
                if phase == "adapt" and not s.adaptation_done:
                    print("WL budget exhausted; saved partial profile. Resume with a larger --steps budget.", flush=True)
                    break
        except KeyboardInterrupt:
            print("Interrupted; saving resumable state.", flush=True)
        finally:
            save_run(out, s, metadata, args, progress)
    print(f"Saved {out} ({time.monotonic()-started:.1f}s).", flush=True)


def parser():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("command", choices=("inspect", "run"))
    p.add_argument("--mem-checkpoint", default=str(base / "Skeletons/memorization_phase.pth"))
    p.add_argument("--gen-checkpoint", default=str(base / "Skeletons/training_checkpoint.pth"))
    p.add_argument("--n-heads", type=int, default=4, help="Cannot be inferred from weight shapes.")
    p.add_argument("--train-frac", type=float, default=.3)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    target = p.add_mutually_exclusive_group()
    target.add_argument("--x-star", type=float, help="Natural log of target training CE; default: measured mem checkpoint.")
    target.add_argument("--loss-star", type=float, help="Target raw training CE; internally converted to ln(loss).")
    p.add_argument("--half-width", type=float, default=.1, help="Shell half-width in natural-log loss units.")
    p.add_argument("--no-match-loss", action="store_true", help="Disable initialization by output-logit scaling.")
    p.add_argument("--start", choices=("mem", "gen"), default="mem")
    p.add_argument("--box-factor", type=float, default=3., help="B=factor*largest absolute prepared endpoint weight.")
    p.add_argument("--box-bound", type=float, help="Explicit shared parameter box half-width B.")
    p.add_argument("--y-min", type=float, default=.1)
    p.add_argument("--y-max", type=float, default=.995)
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--step-size", type=float, default=.001, help="Fixed Gaussian std per selected parameter (absolute units).")
    p.add_argument("--block-size", type=int, default=256, help="Number of random coordinates per move; 0=all.")
    p.add_argument("--steps", type=int, default=100000, help="Total WL step budget, also on resume.")
    p.add_argument("--burn-in", type=int, default=5000, help="Discarded moves after freezing bias.")
    p.add_argument("--production-steps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log-f", type=float, default=1.)
    p.add_argument("--log-f-min", type=float, default=.001)
    p.add_argument("--flatness", type=float, default=.8)
    p.add_argument("--min-visits", type=int, default=20)
    p.add_argument("--check-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--left-phase", type=float, nargs=2, default=[.10, .30], metavar=("LOW", "HIGH"))
    p.add_argument("--right-phase", type=float, nargs=2, default=[.90, .995], metavar=("LOW", "HIGH"))
    p.add_argument("--output", default="entropy_runs/mem_seed1")
    p.add_argument("--resume", help="Saved sampler.pt; repeat original settings, increase total step budgets.")
    return p


def main():
    p = parser()
    args = p.parse_args()
    numeric = [v for v in vars(args).values() if isinstance(v, float)] + args.left_phase + args.right_phase
    if not all(math.isfinite(v) for v in numeric):
        p.error("Numeric arguments must be finite.")
    if not (0 < args.train_frac < 1 and 0 <= args.y_min < args.y_max <= 1):
        p.error("Require 0<train-frac<1 and 0<=y-min<y-max<=1.")
    if not (0 < args.log_f_min < args.log_f and 0 < args.flatness <= 1):
        p.error("Require 0<log-f-min<log-f and 0<flatness<=1.")
    positive = ("half_width", "box_factor", "step_size", "batch_size", "threads", "bins", "steps",
                "min_visits", "check_every", "log_every", "save_every", "n_heads")
    if any(getattr(args, k) <= 0 for k in positive) or args.bins < 3:
        p.error("Sizes, intervals and scales must be positive; bins must be >=3.")
    if args.block_size < 0 or args.burn_in < 0 or args.production_steps < 0:
        p.error("Block size, burn-in and production steps must be nonnegative.")
    if (args.loss_star is not None and args.loss_star <= 0) or (args.box_bound is not None and args.box_bound <= 0):
        p.error("loss-star and box-bound must be positive.")
    if not (args.left_phase[0] < args.left_phase[1] < args.right_phase[0] < args.right_phase[1]):
        p.error("Phase intervals must be ordered and disjoint.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    try:
        if args.command == "inspect":
            _, _, metadata = setup(args)
            print(json.dumps(metadata, indent=2))
            print("Verify n-heads, train-frac and data-seed: old checkpoints do not store them.")
        else:
            run(args)
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
