"""GTA Mode — strict, signal-driven clipping of famous streamers' VODs and lives.

A mode of its own, separate from the normal pipeline: links are queued, only
the audio of each stream is downloaded for analysis, a Hype Curve (audio
spikes, chat velocity and keywords, YouTube "Most Replayed", Twitch viewer
clips) steers the transcript read in 12-minute chunks, every moment goes
through a harsh two-phase judgement, and only the survivors' sections are
downloaded in full quality and rendered uncropped (whole stream frame over a
blurred fill, hook title on top, word-by-word captions below).
"""
