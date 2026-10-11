# Presentation script and defense guide

*Group 3 · Bisaya post-class speech and self-reported affect · 40 participants · audio: CS2B_v2_wav_16k. Built by `src/10b_build_presentation_script.py` from the result files, so every number matches the slides and the report.*

## How to use this

- **Part 1** is what to say, slide by slide. It takes about 12 minutes.
- **Part 2** explains every idea in plain words. Read it once before you present.
- **Part 3** has the questions you are most likely to get, with honest answers.
- **Part 4** is the short list of numbers to remember.
- **Part 5** lists what not to say, and what to say instead.
- Don't read the script word for word. Learn the first sentence of each slide and the number you point at.
- For four speakers, split the slides 1–3, 4–7, 8–10 and 11–13.
- If you are given only five minutes, present slides 2, 3, 8, 10 and 12 and skip the rest.
- The speaker notes under each slide in PowerPoint are the same words as Part 1.

## The whole story in one minute

> We asked 40 second-year CS students to say one Bisaya word right after class and to rate how they felt. We measured 108 properties of each voice clip and trained three kinds of models to guess the feeling from the voice alone, always testing on students the model had never heard.
>
> The models did not beat guessing once we corrected for running 9 tests. We also found a trap: the word people choose is itself linked to how they feel, and the computer can tell which word was said. So a model can look smart by recognising the word, not the emotion.
>
> Our honest conclusion: with 40 noisy one-word recordings, our data do not show that post-class feelings can be recognised from the voice. What we built is a complete, tested pipeline that re-runs everything with one command when better data arrives.

## Part 1 · What to say, slide by slide

### Slide 1 · Title

**Say** (about 20 seconds)

> Good day. We are Group 3. Our project asks one simple question: can a single spoken word, said right after class, tell us how a student feels?
>
> We studied second-year Computer Science students. We will show what we collected, how we analysed it, what we found, and what we could not show.

**Point at:** Nothing yet. Look at the audience.

### Slide 2 · What we asked

**Say** (about 60 seconds)

> Right after class, each participant said one Bisaya word that matched their reaction, for example Kapoy, Salamat or Lingaw. Then they rated how they felt on two scales from 1 to 5.
>
> Valence is how pleasant the feeling is. Arousal is how much energy it has: calm or tired at the low end, excited or tense at the high end.
>
> Our question: can the sound of the word, not its meaning, tell us those ratings? From each recording the computer took 108 voice measurements.
>
> And we made the test strict. The computer is always tested on people it has never heard, because a real system would meet new students.

**Point at:** The three tiles on the right: 1 word, 2 ratings, 108 measurements.

**In plain words:** The correct answer is the participant's own rating. We never decide the feeling from the meaning of the word. Someone can say *Kapoy* and still report feeling fine. The technical name for the strict test is *speaker-independent*.

**If asked: “Why use self-report as the correct answer?”** The protocol defines it that way. The feeling belongs to the participant. A listener's impression, or the meaning of the word, would be our own assumption.

### Slide 3 · The data we had

**Say** (about 75 seconds)

> The sheet lists 40 participants and we analysed all 40, which meets the protocol minimum of 40.
>
> We want to be open about the audio. We recorded phone videos, and a group member converted them to short 16 kHz WAV files. Those files are what we analysed. We checked every file against the original recordings. 31 match an original. 9 could not be checked, and 3 of those show signs of editing, such as stretches of exact silence. We kept them, marked them, and ran the models again without them. You will see that result later.
>
> 9 spoken words are blank in the sheet. We never guessed a word from the feeling text; those files carry the label UNVERIFIED.
>
> The recordings were made in a hallway right after class, so they are noisy: we detected background voices in 16 of 40.
>
> And consent: the sheet says Yes for everyone, but the log of the signed forms still has to be completed.

**Point at:** The four tiles, then the first bullet about the audio.

**In plain words:** *Could not be checked* means we have no original recording to compare with, so we cannot prove when the file was recorded or that it is unedited. *Exact silence* is a stretch of perfect zeros. A real phone recording always contains a little noise, so perfect zeros mean the file was edited.

**If asked: “Why did you analyse copies and not the original recordings?”** We had original recordings for only 32 of the 40 participants. The version-2 set covers all 40, so the group decided to use it for every requirement. We recorded that decision, kept the originals untouched, audited every file against them, and report the results with and without the uncertain files.

**If asked: “Did you use the denoised files?”** No. The protocol forbids denoising, and denoising changes exactly the properties we measure. We analysed only the non-denoised set, and the pipeline passes those files through sample for sample.

### Slide 4 · How we analysed it

**Say** (about 70 seconds)

> Everything runs with one command, in eight steps.
>
> Check: every file and every sheet cell is checked, and nothing is changed. Clean: fixes are made by fixed rules only, and each fix is logged. Find the word: the computer finds the spoken word inside each clip without ever seeing the ratings. Measure: 108 numbers for each voice.
>
> Then we explore the data, train the models, test whether the scores are better than luck, and run the word check, which asks whether the model hears the feeling or only the word.
>
> The three rules at the bottom keep the test fair. No participant is in both the training and the test data. The model's settings are learned from the training data only. And the ratings are never used to cut or choose audio.

**Point at:** Walk along the eight boxes with your hand, then point at the three rules.

**In plain words:** The three rules prevent *data leakage*: the model must not see the answers before the exam. If we prepared the measurements using all the data, a little information about the test participants would leak into training.

**If asked: “What are the 108 features?”** MFCC (26); delta MFCC (26); delta-delta MFCC (26); F0 (pitch) (9); energy (6); zero-crossing rate (6); spectral (6); duration/voicing (3). Pitch is how high the voice is, energy is loudness, and the MFCCs describe the shape of the sound spectrum, which is what makes voices and vowels sound different.

### Slide 5 · What the participants reported

**Say** (about 45 seconds)

> Most participants reported neutral to pleasant feelings and moderate to high energy. The two ratings rise together: the correlation is 0.71.
>
> Some scores were chosen by very few people, so we cannot model all five scores. We grouped them, and we fixed the groups before modelling: arousal into low, moderate and high; valence into pleasant and not pleasant; and arousal into high and not high.
>
> We modelled a grouping only if every group has at least five people.

**Point at:** The two bar charts, then the groupings on the right.

**In plain words:** Valence 3–5: 37 of 40. Arousal 3–5: 30 of 39. Group sizes: arousal_3: Low: 9 / Moderate: 13 / High: 17; valence_2: NotPleasant: 18 / Pleasant: 22; arousal_2: NotHigh: 22 / High: 17. Arousal has one person fewer because Y2-076's arousal rating is crossed out in the sheet, so we did not use it. In the chart, the blue and orange bars are the same height because every participant in the sheet is analysed.

**If asked: “Why not predict all 25 combinations of valence and arousal?”** With 40 people, most of the 25 cells would be empty or hold one person. A model cannot learn or be tested on a group with one example.

**If asked: “Did you invent these groups?”** No. They are groupings of the participants' own 1-to-5 ratings. The original ratings stay in the data, and the groupings are marked as derived.

### Slide 6 · Simple voice measures show no clear link

**Say** (about 60 seconds)

> Before any machine learning, we checked simple links. We chose six well-known voice features in advance: pitch level, pitch range, loudness, duration, brightness and the share of voiced sound.
>
> None of them is significantly linked to the ratings. The strongest are brightness (spectral centroid) with arousal at +0.26 and loudness (RMS) with valence at +0.26. Those are weak.
>
> Across all 108 features, 0 pass the correction for testing many features at once for arousal, and 0 for valence.
>
> With 40 people, only links of about 0.44 or stronger can be found reliably. A moderate link of 0.3 would need about 90 participants.

**Point at:** The table, then the last bullet about sample size.

**In plain words:** The symbol ρ (rho) is the Spearman correlation: +1 means the two rise together perfectly, 0 means no link, −1 means one rises as the other falls. We use Spearman because the ratings are ranks from 1 to 5.

**If asked: “Why choose six features in advance?”** If you test many features and report only the best ones, you will always find something by luck. Choosing in advance keeps us honest. We still report all of them, with a correction for testing many features.

### Slide 7 · How to read our results

**Say** (about 50 seconds)

> Before the results, five ideas.
>
> Chance is the score you get by pure guessing: 0.33 with 3 groups, 0.50 with 2 groups.
>
> Balanced accuracy is the average of the scores for each group. A model cannot look good by always giving the most common answer.
>
> The 95% interval is the range where the true score probably lies. Ours are wide, because 40 people is few.
>
> The p-value tells us how often shuffled ratings, which mean nothing, score as high as the real ones. A small p-value means the score is unlikely to be luck.
>
> And the correction. We ran 9 tests, and with 9 tries a lucky result is expected. The Holm correction accounts for that.

**Point at:** Each row, as you explain it.

**If asked: “Why is the limit 0.05?”** It is the usual limit in research. It means we accept a 5% risk of calling luck a real effect.

### Slide 8 · The models did not beat chance

**Say** (about 70 seconds)

> Now the main test. For each target we show the best of three models.
>
> The random forest was best: 0.48 for three-level arousal (low, moderate, high), where chance is 0.33; 0.59 for valence (pleasant vs not pleasant), where chance is 0.50; 0.56 for two-level arousal (high vs not high), where chance is 0.50.
>
> Is that better than luck? We shuffled the ratings 500 times and ran the whole procedure again each time. 1 of 9 tests has a p-value below 0.05: the smallest is 0.022, for the random forest on three-level arousal (low, moderate, high).
>
> But we ran 9 tests. With 9 tries, about 0.45 lucky results are expected. After the Holm correction that p-value becomes 0.198, which is not significant. So no model stays significant after the correction.

**Point at:** The p-value column, then the column *p after correction*.

**In plain words:** The *Chance* column is what a model scores if it always gives the same answer. Holm in one line: multiply the smallest p-value by the number of tests. 0.0220 × 9 = 0.198.

**If asked: “Isn't 0.48 against 0.33 a good result?”** It looks promising, but three things stop us from claiming it. The 95% interval runs from 0.35 to 0.62, so the low end is close to chance. It does not survive the correction for 9 tests. And without the 9 files that could not be checked, it falls to 0.37.

**If asked: “That interval starts at 0.35, which is above 0.33. Doesn't that mean it beats chance?”** Only just, and only for this one test. It says the same thing as the p-value of 0.022 before correction. But it is one of 9 tests, and with that many tests about 0.45 lucky results are expected. After the correction it is not significant, and it drops to 0.37 when the unchecked files are left out.

**If asked: “Why three models? Which is best?”** A linear model (logistic regression), a kernel model (SVM) and a tree ensemble (random forest) make different assumptions, so together they are a fair test. The random forest scored highest, but only 1 of the pairwise comparisons has p below 0.05, uncorrected, so we do not claim one model is truly better.

### Slide 9 · The same result as a picture

**Say** (about 40 seconds)

> This figure shows the same result as a picture, with one panel for each target. Each dot is a model's balanced accuracy. The horizontal bar is its 95% interval. The vertical line is chance.
>
> The grey rows are baselines that never listen to the audio. The blue rows are our models.
>
> The bars are wide. 8 of the 9 blue bars cross the chance line. The one that just clears it is the random forest for three-level arousal, and that is the result that did not survive the correction. This is what an inconclusive result looks like.

**Point at:** One blue bar that crosses the vertical line.

**In plain words:** A wide bar means the score is uncertain. If the bar crosses the chance line, the true score could be no better than guessing.

### Slide 10 · Is it the voice, or just the word?

**Say** (about 75 seconds)

> This is our most important finding. Is the model hearing the feeling, or only recognising the word?
>
> First test: can the computer tell Kapoy from the other words by sound alone? Yes. Balanced accuracy 0.85, with p = 0.002.
>
> Second test: is the word linked to the feeling? Yes. Of 10 Kapoy speakers, 8 felt not pleasant. Of 21 speakers of other words, only 5 did.
>
> So a model can score above chance by recognising Kapoy and answering *not pleasant*, without hearing any feeling at all.
>
> When we look at Kapoy speakers alone, the same word with different feelings, 0 of 12 links between voice and rating are significant.

**Point at:** The small table: Kapoy 8 and 2, other words 5 and 16.

**In plain words:** This is called a *confound*: a hidden third thing that explains the link. Here the word explains both the sound and the feeling. It is like concluding that umbrellas cause rain because they appear together.

**If asked: “How would you fix the word problem?”** Ask every participant to say the same word, or collect enough recordings of each word to analyse each word separately. Both are in our plan for a larger study.

**If asked: “Why only Kapoy?”** It is the only word said by enough people (10). Every other word was said by three people or fewer, which is too few to analyse.

### Slide 11 · Limits of this study

**Say** (about 55 seconds)

> We want to be clear about the limits. The sample is small. With 40 people only strong links can be found.
>
> The audio is converted copies, not the original recordings: 9 could not be checked and 3 show signs of editing.
>
> The recordings are noisy: we detected background voices in 16 of 40, in a hallway right after class. And 5 files come from recordings made with a different audio setting, although that setting shows no significant difference here.
>
> Each person said only one word, so we cannot separate a person's normal voice from their feeling at that moment.
>
> Everyone came from the same year level, class activity and session, so we cannot compare contexts. And we measured links, not causes.

**Point at:** The first three bullets.

**In plain words:** *Power* is the chance of finding an effect that really exists. With 40 participants our study can only find links of about 0.44 or more. *A different audio setting* (5 files) means the original was saved in stereo with a second audio track, unlike the others.

### Slide 12 · Conclusion

**Say** (about 45 seconds)

> Our conclusion: our data do not show that post-class feelings can be recognised from the voice.
>
> That is a real result, not a failure. A careful analysis that finds no reliable signal is more useful than an impressive number that would not hold up.
>
> We contribute three things. A complete, tested pipeline that repeats the whole analysis with one command. The finding that the chosen word is linked to the feeling and must be controlled. And a clear statement of how much data a real study needs: about 90 participants.

**Point at:** The blue sentence.

### Slide 13 · Next steps

**Say** (about 30 seconds)

> Our next steps are concrete. Complete the consent log from the signed forms. Have two group members listen to every clip without seeing the ratings. Find the original recordings for the files we could not check. Record more participants with one device, one setting and one take. Follow the analysis plan we wrote before seeing new data. And confirm the open protocol questions with our instructor.
>
> Thank you. We are happy to take your questions.

**Point at:** The audience.

## Part 2 · Every idea in plain words

### Valence and arousal

- **What it is:** two ways to describe a feeling. Valence is unpleasant to pleasant. Arousal is low energy to high energy.
- **Examples:** tired is low arousal; excited is high arousal and pleasant; stressed is high arousal and unpleasant.
- **In our project:** each participant rated both from 1 to 5. The two ratings rise together (ρ = 0.71).

### Groups (derived classes)

- **What it is:** groups made from the 1-to-5 ratings, because some single scores have too few people. The slides say *group*; the technical word is *class*.
- **The rule:** a grouping is modelled only if every group has at least 5 people.
- **In our project:** `arousal_3` (Low: 9 / Moderate: 13 / High: 17); `valence_2` (NotPleasant: 18 / Pleasant: 22); `arousal_2` (NotHigh: 22 / High: 17). Not modelled: `valence_5`, `arousal_5`, `valence_3`.

### Acoustic features

- **What it is:** numbers that describe a sound, measured by the computer.
- **Pitch (F0):** how high or low the voice is.
- **Energy (RMS):** how loud it is.
- **Zero-crossing rate and spectral centroid:** how noisy or bright the sound is.
- **MFCCs:** 13 numbers that describe the shape of the sound spectrum. The *delta* versions describe how fast they change. For each we keep the average and the spread over the word, so 13 × 2 = 26 per group.
- **In our project:** 108 features per recording, measured only on the located word: 26 + 26 + 26 + 9 + 6 + 6 + 6 + 3 = 108.
- **A caution about pitch:** pitch in Hz mostly shows whether a person's voice is naturally high or low. With one recording per person we cannot subtract their normal voice.
- **Tools:** Python, with Praat (through parselmouth) for pitch, librosa for the spectrum and MFCCs, and scikit-learn for the models.

### Finding the word in the clip

- **What it is:** each clip also contains hallway noise, so the computer looks for the loudest short event and treats that as the word.
- **Why it matters:** it never uses the labels, so it cannot cheat.
- **In our project:** 13 of 40 recordings are flagged as uncertain, for example because the word touches the edge of the file. Nobody has yet checked them by listening.

### Speaker-independent testing and cross-validation

- **What it is:** we split the participants into 5 groups, train on 4 and test on the fifth, and rotate so everyone is tested once.
- **Like:** practising with four chapters and being examined on the fifth, five times.
- **Repeats:** we repeat this 10 times with different splits, so one lucky split cannot decide the result.
- **Nested:** inside each training set there is another small split used only to choose the model's settings. The test group is never used for that.
- **Speaker-independent:** nobody is in both the training and the test group. Here each person has one recording, so it holds automatically, and the code checks it anyway.

### Data leakage

- **What it is:** information about the test data slipping into training. It makes results look better than they are.
- **How we prevent it:** only acoustic features go into the model; scaling and filling of missing values are fitted on the training folds only; the code stops if a non-acoustic column appears.

### The models and the baselines

- **Logistic regression:** draws straight boundaries between classes. The simplest of the three.
- **SVM:** can draw curved boundaries.
- **Random forest:** 300 decision trees that vote.
- **Dummy baselines:** models that never look at the audio. One always answers the most common class; one guesses at random in the same proportions as the classes. A real model must beat them.
- **Class weights:** each class counts equally even when it has fewer people. We did not create artificial samples (no SMOTE).

### Balanced accuracy

- **What it is:** the average of the scores for each class.
- **Why not plain accuracy:** in our three-level arousal task, always answering the most common class is right 44% of the time without learning anything. Balanced accuracy gives that lazy model exactly the chance level, 0.33.
- **Chance level:** 1 ÷ number of classes. 0.33 for three classes, 0.50 for two.

### Confidence interval

- **What it is:** the range where the true value probably lies.
- **How we get it:** we re-draw the participants at random, with replacement, 2000 times, and recompute the score each time. The middle 95% of those scores is the interval. This is called a bootstrap.
- **In our project:** the best model's interval is 0.35 to 0.62. That is wide, because 39 people is few.

### p-value and the permutation test

- **What it is:** we shuffle the labels so they no longer belong to the right recordings, then run the whole procedure again. We do that 500 times.
- **The p-value:** the share of shuffles that score as high as the real data. A small p means the real score would be rare by luck.
- **In our project:** the smallest p is 0.022: about 2 in 100 shuffles did as well.

### Multiple testing: Holm and FDR

- **The problem:** if you run many tests, some will look significant by luck. Roll a die many times and a six is no surprise.
- **Holm:** the correction we use for the 9 model tests. The smallest p is multiplied by 9: 0.0220 × 9 = 0.198.
- **FDR (Benjamini–Hochberg):** the correction we use for the 108 feature correlations.
- **Expected false alarms:** 0.05 × 9 = 0.45 tests would pass by luck even if nothing were there.

### The lexical confound

- **What it is:** the word explains both the sound and the feeling, so the model may learn the word.
- **Evidence 1:** the audio identifies Kapoy with balanced accuracy 0.85 (p = 0.002).
- **Evidence 2:** Kapoy speakers felt not pleasant 8 times out of 10; others 5 out of 21 (Fisher p = 0.006).
- **Evidence 3:** on two-level arousal the models are right 68% of the time on Kapoy recordings and 49% on other words.

### Sensitivity analysis

- **What it is:** repeating the analysis without a doubtful part of the data, to see whether the result holds.
- **In our project, three-level arousal (low, moderate, high), random forest** (all data: 0.48):
  - without the recordings whose word location is uncertain: 0.41 (n = 27)
  - without the copies of recordings made with a different setting: 0.46 (n = 34)
  - without the edited or clipped files: 0.44 (n = 37)
  - without the files that have no verified original: 0.37 (n = 31)
- **Valence, random forest** (all data: 0.59): between 0.58 and 0.67 across the same scenarios.
- **What it tells us:** the scores move up and down depending on which files are included. That is a sign the sample is too small for a stable answer.

### Statistical power

- **What it is:** the chance that a study detects an effect that really exists.
- **In our project:** with 40 people, only correlations of about 0.44 or more are detectable 80% of the time. A moderate 0.3 needs about 90 people.

### Why raw audio, not denoised

- The protocol forbids denoising.
- Denoising changes loudness, the spectrum and the MFCCs, which are the things we measure.
- It changes each file by a different amount, so a model could learn *how much a file was cleaned* instead of the feeling.

### Reproducibility

- One command runs every step, with fixed random seeds.
- Automated tests check the rules that decide the data.
- A validation step re-checks the whole project: 24 checks pass, 1 blocked.
- 11 of 12 required outputs are complete. Not complete: Required consent documentation.

## Part 3 · Likely questions and honest answers

**“So your model does not work. What is the point of the project?”**

The point was to find out whether it works, and to find out properly. We built a pipeline that answers the question honestly, and the answer is: not with 40 noisy one-word recordings. We also found out why a naive analysis would be misleading, namely the word confound, and how much data a real study needs.

**“Your best p-value is below 0.05. Why do you say nothing is significant?”**

Because we ran 9 tests. If you run 9 tests on pure noise, about 0.45 of them pass by luck. After the Holm correction, 0.022 becomes 0.198.

**“Why did you not use deep learning?”**

Deep networks need thousands of examples. We have 40. With so little data a large model memorises the training set, so simple models with cross-validation are the right tools.

**“Why did you not use SMOTE or oversampling?”**

With this few people, artificial samples would be copies of the same few voices, and they leak information if created before the split. We used class weights instead, which need no artificial data.

**“Why are some spoken words UNVERIFIED?”**

9 words are blank in the data sheet. The protocol forbids guessing the word from the feeling text, so we left them blank. The word is not a model input, so the models are not affected. Only the word analysis leaves those participants out.

**“Some of your files were edited. Is your data valid?”**

We do not hide that. 3 files contain inserted digital silence and 9 have no verified original. We flagged each one and re-ran the models without them: the main arousal score moves from 0.48 to 0.37 without the unverified files. Our conclusion does not depend on those files. The clean fix is to find the original recordings.

**“How do you know there is no data leakage?”**

Three checks. Only acoustic columns may enter the model, and the code stops otherwise. Scaling and filling of missing values happen inside each training fold. And the code verifies that no participant is in both the training and the test fold.

**“How did you choose which part of the audio is the word?”**

Automatically, from loudness alone: the dominant short event in the clip. The rule never sees the labels. 13 recordings are flagged as uncertain, and a blind listening review by two group members is the next step.

**“Is consent documented?”**

The sheet records Yes for every participant, but the log of the signed forms is not complete yet. That is why the report marks the consent requirement as blocked, and marks its own dissemination as blocked until the log is complete. Completing it is the first of our next steps.

**“Did you account for different voices, for example male and female?”**

We could not. The sheet has no sex or gender column, and each person gave one recording, so there is no personal baseline to subtract. Pitch in Hz therefore mostly reflects the natural voice. That is the *one word per person* limit on slide 11, and it is one reason we do not over-read the pitch features.

**“Who prepared the version-2 audio files, and how?”**

A group member converted the recordings using ChatGPT: each clip was cut to the part around the word and saved as a 16 kHz WAV file. We did not take that conversion on trust. For the 31 files where we have the original, our audit shows that the copy matches the original (similarity 0.97 or higher out of 1), that it is a cut of 1.3 to 2.6 seconds from it, and that the loudness was not changed. For the other 9 we have nothing to compare with, so they are flagged and the models were re-run without them. Before you present, open that conversation and note the steps or code it used, so you can describe them in your own words. The project has no record of them.

**“Your earlier submission had different numbers. Why?”**

The earlier analysis used the original phone recordings, which covered 32 participants. This one uses the version-2 audio for all 40. Different audio and more people give different numbers. The conclusion is the same in both: no model beats chance after correction, and the word confound is present. One detail did change: 3 features passed the correction for arousal before, and none do now.

**“What would you do with more time?”**

Collect about 90 participants with one device and one setting, have everyone say the same word or enough repetitions of each word, verify the word boundaries by listening, and follow the analysis plan we wrote before seeing new data.

**“How was the code written? Do you understand it?”**

Answer this truthfully. The pipeline was implemented with an AI coding assistant under the group's direction; the commit history shows it. Say which decisions were yours (the data, the audio set, what to report), and explain any step using Part 2. Check your course's rule on AI tools before you present.

## Part 4 · Numbers to remember

| What | Number |
|---|---|
| Participants analysed | 40 (protocol minimum 40) |
| Features per recording | 108 |
| Models | 3, plus 2 dummy baselines |
| Cross-validation | 5 folds × 10 repeats, nested |
| Permutations per test | 500 |
| Best score, three-level arousal (low, moderate, high) | 0.48 (chance 0.33), interval 0.35–0.62 |
| Best score, valence (pleasant vs not pleasant) | 0.59 (chance 0.50), interval 0.47–0.71 |
| Best score, two-level arousal (high vs not high) | 0.56 (chance 0.50), interval 0.43–0.69 |
| Smallest p-value | 0.022 → 0.198 after Holm (9 tests) |
| Features passing correction | 0 for arousal, 0 for valence |
| Kapoy recognised from audio | 0.85 balanced accuracy, p = 0.002 |
| Kapoy and feeling | 8 of 10 not pleasant; others 5 of 21 |
| Files without a verified original | 9 (Y2-053, Y2-071, Y2-074, Y2-075, Y2-076, Y2-077, Y2-078, Y2-079, Y2-080) |
| Edited or clipped files | 3 (Y2-076, Y2-077, Y2-078) |
| Words missing in the sheet | 9 |
| Smallest detectable correlation | about 0.44; 0.3 needs about 90 people |
| Validation | 24 pass, 1 blocked (consent) |

## Part 5 · What not to say

| Don't say | Say instead |
|---|---|
| “Our model detects emotions.” | “Our models did not beat chance after correction.” |
| “We got 48% accuracy.” | “Balanced accuracy 0.48 where chance is 0.33, with a wide interval.” |
| “The result is significant.” | “One raw p-value is below 0.05, but it does not survive correction.” |
| “Kapoy means the student is sad.” | “Kapoy speakers more often reported not-pleasant feelings.” |
| “The class caused the feeling.” | “These are associations. The design cannot show causes.” |
| “The data is clean.” | “We flagged every doubtful file and tested the results without them.” |
| “The project failed.” | “The answer to our research question is no, with this data.” |

