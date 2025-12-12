<div>
<h1 style="text-align:center;text-decoration:underline"> AIML - Learning Notes </h1>
</div>

-----

## <u> Logistic Regresssion: </u>

**CountVectorizer()**

**TFIDFVectorizer()**

**ROC - AUC**
- ROC &rarr; Receiver Operating Characteristics
- AUC &rarr; Area Under the curve

**First: what ROC and AUC are not**
	•	They are not accuracy
	•	They are not tied to a single threshold
	•	They do not tell you how many emails go to spam

ROC–AUC is about ranking quality, not final decisions.

⸻

**Step 1: What ROC actually is**

ROC = Receiver Operating Characteristic

Historically:
-	WWII radar operators
- “Receiver” = signal detector
- “Operating characteristic” = trade-off between detecting signals and false alarms

Same math, new domain.

⸻

**Step 2: The two axes of the ROC curve**

ROC plots rates, not raw counts.

Y-axis: True Positive Rate (TPR)

Also called Recall.

\text{TPR} = \frac{TP}{TP + FN}

Meaning:

“Of all real spam emails, how many did I catch?”

⸻

X-axis: False Positive Rate (FPR)

\text{FPR} = \frac{FP}{FP + TN}

Meaning:

“Of all legit emails, how many did I wrongly flag as spam?”

This is the cost axis.

⸻

**Step 3: How the ROC curve is built**

Your model outputs probabilities, not labels:

Email A → 0.97
Email B → 0.83
Email C → 0.12
Email D → 0.03

Now imagine sliding the threshold:

Threshold	Predicted spam	TPR	FPR
1.0	none	0.00	0.00
0.9	only A	low	very low
0.5	A, B	higher	moderate
0.1	A, B, C	high	high
0.0	all	1.00	1.00

Each threshold gives one (FPR, TPR) point.

Plot all of them → you get the ROC curve.

⸻

**Step 4: What the shape means**

Perfect model
	•	Spam always scores higher than non-spam
	•	ROC curve hugs the top-left corner
	•	TPR = 1, FPR = 0

Random model
	•	No separation
	•	ROC is the diagonal line
	•	TPR = FPR at all points

Bad model
	•	Below diagonal
	•	Means it’s ranking spam worse than normal
	•	Flip predictions → suddenly it’s good

⸻

**Step 5: What AUC actually means**

AUC = Area Under the ROC Curve

But the interpretation is the key:

AUC = probability that the model ranks a random positive higher than a random negative

Formally:

\text{AUC} = P(\text{score(spam)} > \text{score(normal)})

Examples:
	•	AUC = 0.5 → random guessing
	•	AUC = 0.75 → 75% of the time, spam outranks normal
	•	AUC = 1.0 → perfect ranking
	•	AUC = 0.3 → confidently wrong

This is a ranking metric, not a decision metric.

⸻

**Step 6: Why ROC–AUC ignores thresholds**

ROC–AUC evaluates all thresholds simultaneously.

That’s why it’s:
	•	Great for comparing models
	•	Terrible for choosing operating points

It tells you:

“Is this model capable of separating classes?”

Not:

“Where should I set my spam filter?”

⸻

Step 7: Why ROC–AUC can lie (important)

ROC–AUC treats:
	•	FP cost
	•	FN cost

as equal.

In spam filtering, that’s false.

Example:
	•	False positive = miss a bank alert (very expensive)
	•	False negative = see a spam email (annoying)

ROC does not know or care.

This is why high ROC–AUC models can still be unusable in production.

⸻

Step 8: ROC vs Precision–Recall (why this matters)

When data is imbalanced (spam ≪ normal):
	•	ROC–AUC can look great
	•	Precision can still be garbage

Because FPR can be tiny even when FP count is large.

That’s why:
	•	ROC–AUC → model quality
	•	Precision–Recall → operational quality

⸻

Bringing it back to your spam classifier

When you compute:

auc = roc_auc_score(y_test, y_proba)

You are asking:

“Does my TF-IDF + Logistic Regression rank spam above non-spam?”

If AUC is:
	•	< 0.6 → model barely learns anything
	•	0.7–0.85 → decent classical NLP model
	•	> 0.9 → suspiciously good (often data leakage)

But that alone does not tell you:
	•	How many legit emails you’ll lose
	•	Where to set the threshold
	•	Whether users will complain

⸻

Mental model to keep
	•	ROC curve = how ranking quality degrades as you loosen your filter
	•	AUC = how good the ranking is overall
	•	Threshold = business decision, not a metric

If you want next:
	•	Why PR-AUC is often better than ROC-AUC for spam
	•	How to pick thresholds using cost curves
	•	How ROC-AUC relates to Mann–Whitney U statistics

Those are the places where this stops being theory and starts driving real systems.