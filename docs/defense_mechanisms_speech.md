
## English Script

I'll introduce four defense methods for LLM-as-Judge.

---

### Part 1 — Four Defense Methods

First, **Self-Reminder:** We prepend a short safety reminder before the grading prompt, telling the model to ignore post-answer injections and follow the original rubric.

Next, **Paraphrase Defense:** We wrap the input in a rewrite-then-grade instruction—the model implicitly strips adversarial tokens before scoring.

**Hijacking Suppression** is an inference-time hook that down-weights attention from adversarial suffixes by a factor beta, then re-normalizes. It counters **attention hijacking** attacks like GCG.

**Attention Sharpening** also runs at inference time. Before the softmax, it divides logits by a temperature below one, sharpening the distribution. It targets **attention slipping**—when focus diffuses and adversarial content slips through.

---

### Part 2 — Comparison

These four defenses differ along three axes.

First is the **intervention level**: SR and PD are **prompt-level**, just add extra texts; HS and AS are **mechanism-level**, modifying attention computation inside the model.

Second, for **deployment cost**: SR is cheapest—just one forward pass, no hooks. PD adds prompt length but still no model changes. HS and AS are heavier. They require **model hooks** and eager attention; HS also needs suffix-length information.


---

### Part 3 — Experimental Results

For the experiment part, we tested all four against RolePlay, GCG, and injection attacks on SciEntsBank with Llama-3.1-8B-Instruct. We track **Attack Success Rate** for defense effect and **Quadratic Weighted Kappa** for grading quality—a good defense lowers ASR while keeping QWK intact.


For **SR**, On binary RolePlay samples, ASR dropped to zero, with 107% QWK retention. Against GCG samples, ASR only fell from 99% to 95%, with 102% QWK retention.

For PD, it's the strongest defense against GCG—ASR fell from 99% to **36%**, with 137% QWK retention. On RolePlay, ASR reached zero, but QWK collapsed a lot.

For HS, it's the best mechanistic defense on GCG—ASR from 100% to **75%**. It also helped DC injection: 74% down to 54%. But against AO and IM, ASR stayed at 100%, which means no effect.

For AS, it underperformed HS across the board. GCG defended ASR was still 98%. On DC injection, ASR even rose from 74% to 78%. 

Mechanistic analysis may explains this result: GCG suffixes capture about 14% of model attention via hijacking; RolePlay captures only about 4% and succeeds through semantic persuasion. Defenses aligned with the attack mechanism work; mismatched ones fail.

---

### Part 4 — Conclusion

Here comes the conclusion.

No single defense wins everywhere. SR is the best low-cost choice for semantic attacks, preserving grading quality. PD is the go-to against GCG, but watch for QWK side effects. HS is the strongest inference-time option when attention hijacking is the threat, but its  deployment costs a lot. For AS, in our current setup, it has limited practical value.

The broader lesson is: **defense must match attack mechanism**—prompt-level guards for semantic injection; attention-level guards for token-level hijacking.
