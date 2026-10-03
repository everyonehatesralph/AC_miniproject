# Draft: e-mail to the instructor

*Draft only. Review it, adjust the wording, and send it yourselves.*

---

**Subject:** CS412 mini-project, Group 3: seven questions about the protocol before final submission

Good day, Sir/Ma'am,

Group 3 (Second-Year CS) has finished processing its data. Before we finalise the report, we would like your
rulings on seven points where the protocol leaves room for interpretation. For each one we state what we
currently assume, so the analysis can be changed easily if you rule otherwise.

1. **Spoken words outside the list.** Section VIII lists seven words (Salamat, Ambot, Grabe, Lisod, Sayon,
   Lingaw, Kapoy) but also says "any response is acceptable". 16 of our 31 recorded words are not on the list
   (for example Nalipay, Katulgon, Gutom, Okay).
   *Current assumption:* they are acceptable, because the word is not a model input. Should they stay in the
   dataset, or be excluded or analysed separately?

2. **Sample size.** We have 40 participants in the sheet but original recordings for only 32. We are still
   looking for the other 8 (see question 7).
   *Question:* may we collect additional consented participants (same protocol, new codes Y2-081 onward,
   coordinated with Group 4) to reach at least 40, or should we report n = 32?

3. **Class activity category.** Our sheet records "Lecture".
   *Current assumption:* it means the protocol category "Lecture/Discussion". Is that correct?

4. **Session label in file names.** Section XII only gives the example `Y2-017_Y2_PM_Grabe.wav`.
   *Current assumption:* `PM` = Afternoon–Evening and `AM` = Morning. Is that correct? What exact time
   boundary should separate the two sessions? Our recordings were made at about 18:10–18:40.

5. **Consent documentation.** What exactly should we submit? Is a de-identified consent log enough (participant
   code, date, collector, form version, consent to voice recording), with the signed forms kept by the group?
   Or do you need the forms themselves?

6. **Incomplete self-reports.** Two participants gave valence/arousal ratings but no feeling description.
   Neither has an original recording, so they are not in the analysis. Should such rows count as valid
   observations?

7. **Converted copies instead of original recordings.** For the 8 participants without original recordings, a
   group member later supplied trimmed 16 kHz WAV copies. Three of them contain inserted digital silence, and
   two cut off the end of the word.
   *Current assumption:* only original, unedited recordings count as observations, so these 8 stay out until
   the originals are found. Is that correct, or may unedited trimmed copies count if the originals are lost?

For your information: our recordings contain heavy background noise. The analysis therefore locates the
participant's word automatically, and a group member will verify each one by listening. The machine-learning
results are currently not significantly better than chance after correction for multiple tests, and we report
this as it is.

Thank you very much.

Respectfully,
Group 3
