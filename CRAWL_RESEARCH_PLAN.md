# Locked Open-Corpus Acquisition Pilot

Protocol locked before observing pilot quality, deduplication, diversity, or ASR results.

- Source: OpenSLR SLR31 Mini LibriSpeech, `dev-clean-2.tar.gz` only.
- Licensing acceptance rule: the official source page must state `CC BY 4.0`.
- Acquisition: one static source page; no browser automation; obey live `robots.txt` and
  the greater of its crawl delay or five seconds on every host.
- Size boundary: at most one 600 MB archive and the first 250 FLAC members in sorted
  archive-path order. This deterministic cap is not selected using audio quality.
- Quality and deduplication: use the repository's unchanged default policy. Exact PCM
  duplicates are removed. Unvalidated MFCC-LSH pairs remain review candidates and are
  not automatically removed.
- Funnel: report pages requested, unique archive candidates, archives attempted and
  downloaded, archive audio members, clips extracted with aligned transcripts, clips
  passing quality, exact duplicates removed, and final clips retained.
- Diversity: compare the acquired manifest before/after curation. Do not infer gender,
  age, or accent from voices; report those fields as unavailable. Report observed
  speaker, linguistic, and duration measures.
- Downstream boundary: export the filtered manifest in the shared `id/path/sentence`
  contract. A frozen-ASR WER audit is descriptive because this read-speech corpus is not
  financial-domain training data. No model improvement claim is permitted without a
  separately locked, multi-seed training intervention and paired 95% confidence interval.
- Claim rule: a null or adverse result is retained. No thresholds or sample cap are tuned
  after observing this pilot.
