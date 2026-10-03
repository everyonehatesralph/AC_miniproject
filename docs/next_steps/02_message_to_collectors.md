# Draft: message to the group members who collected the data

*Draft only. Send it in your group chat. Every answer goes into a file that the pipeline reads, so please
reply precisely. Details behind items 1–4: `reports/supplied_audio_audit.md`.*

---

Hi everyone! The analysis pipeline is finished and on our GitHub. Thank you for the CS2B_v2 files: they helped
us check the data. To make the data precise, we still need these answers from whoever recorded each
participant.

**1. The original recordings for the 8 missing participants (most important).**
- The v2 zip has 16 kHz WAV copies for **Y2-053, Y2-071, Y2-074, Y2-076, Y2-077, Y2-078, Y2-079, Y2-080**,
  but they are trimmed copies, not the phone recordings. Some were also edited or damaged:
  - Y2-076, Y2-077 and Y2-078 contain inserted digital silence;
  - Y2-077 is clipped;
  - Y2-078 and Y2-079 cut off the end of the word.
- Copies like these cannot count as observations. Please find the **original files** (phone gallery, including
  Recently Deleted; videos from 1 Oct, about 18:00–19:00) and upload them to the CS2B Drive folder, named with
  the code (for example `Y2-053.MOV`), **without trimming, denoising or any editing**.
- With these we reach the minimum of 40.

**2. How the v2 files were made.** Who made them, and with which tool? Besides converting to 16 kHz and
trimming, was anything else done, such as removing silence or changing the volume? Please don't use the
*denoised* versions for anything: the protocol forbids denoising.

**3. Y2-075.** The v2 file for Y2-075 is a **different recording** from the Y2-075 video on our Drive (they
don't match). Which one is Y2-075's real post-class recording, and where does the other one come from?

**4. The spoken words.** The words in the v2 file names don't always match the sheet:
- they differ for Y2-046 (Tsadahan vs Salamat), Y2-056 (Libog vs Liboga), Y2-059 (Hangak vs Kapoy) and Y2-061
  (Malipayon vs Malipay);
- for **Y2-066 and Y2-072 they look swapped** (Nalipay and Salamat);
- they also give words where the sheet is blank: Y2-053, 071, 073, 074, 076, 077, 078, 079, 080.

Where do these words come from? Only a word written down at the session, or confirmed by listening, can be
used. A word must never be filled in from the participant's feeling: for Y2-073, Y2-074 and Y2-076 the file name
uses exactly their feeling text.

**5. The 6 stereo recordings.** The files for **Y2-050, Y2-063, Y2-068, Y2-069, Y2-070, Y2-075** were made
with different audio settings, and five of them are time-stamped **2 Oct, 03:11–03:26**, about 9 hours after
the session. Who made them, and how? Were they recorded again later, or exported or edited from the original
videos? If you still have the originals from 1 Oct, please upload them too.

**6. Retakes.** **Y2-044, Y2-047, Y2-056, Y2-057, Y2-064** each have 2–3 takes recorded seconds apart.
- Why was each one retaken? For example: recording started late, noise, the participant laughed, or we asked
  them to repeat. (The protocol says not to ask for repeats until it "sounds emotional", so we need to know.)
- The v2 copies always use the *latest* take. Was that a deliberate choice, and why?

**7. Y2-076.** The arousal value "5 — Very High" is crossed out in the sheet. What did the participant actually
answer? Is there a paper note?

**8. Date and collector.** For each code you collected, tell us the date and your initials. The sheet's Date
Collected and Collector Initials columns are empty.

**9. Consent forms.** Where are the signed consent forms? Please fill in `consent/consent_log_TEMPLATE.csv`
(codes only, **no names**) and save it as `consent/consent_log.csv`. Keep the signed forms themselves
somewhere safe, **not** in the repository.

**10. Listening check (needs a Bisaya speaker; two people is even better).**
1. Run the pipeline once.
2. Double-click `data/review/listening_review.html`, listen to each clip and answer on the page.
3. Save your answers into `data/review/answers/`.

Instructions are on the page and in `docs/RUNNING.md` §6.1. It takes about 1 hour. Please **don't** look at the
sheet (words, feelings, ratings) while listening.

Thank you, team! 🙏
