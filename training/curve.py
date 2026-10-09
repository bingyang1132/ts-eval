"""Pull the held-out curve out of a GRPO log: step, train reward, eval reward/top1/legal.
    python training/curve.py grpo.log
"""
import ast, re, sys

log = open(sys.argv[1]).read().replace('\r', '\n')
train, evals = {}, []
step = 0
for m in re.finditer(r"\{'(?:loss|eval_loss)'.*?\}", log):
    try:
        d = ast.literal_eval(m.group(0))
    except Exception:
        continue
    if 'eval_loss' in d:
        evals.append((step, float(d.get('eval_rewards/oracle_value/mean', 'nan')),
                      float(d.get('eval_rewards/top1/mean', 'nan')), float(d.get('eval_rewards/legal/mean', 'nan')),
                      float(d.get('eval_completions/mean_length', 'nan'))))
    else:
        step += 1
        train[step] = (float(d.get('rewards/oracle_value/mean', 'nan')), float(d.get('rewards/top1/mean', 'nan')),
                       float(d.get('rewards/legal/mean', 'nan')), float(d.get('completions/mean_length', 'nan')),
                       float(d.get('grad_norm', 'nan')))
print('| step | eval oracle_value | eval top1 | eval legal | eval len | train oracle_value (last 25 avg) |')
print('|---|---|---|---|---|---|')
for s, ov, t1, lg, ln in evals:
    window = [train[k][0] for k in range(max(1, s - 24), s + 1) if k in train]
    avg = sum(window) / len(window) if window else float('nan')
    print(f'| {s} | {ov:.3f} | {t1:.3f} | {lg:.3f} | {ln:.0f} | {avg:.3f} |')
print(f'\n{len(train)} train steps logged; last train reward {train[max(train)][0]:.3f}' if train else 'no train steps yet')
