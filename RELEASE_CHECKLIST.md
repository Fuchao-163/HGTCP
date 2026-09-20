# Release checklist

- [ ] Choose an OSI-approved license, add `LICENSE`, and record its SPDX ID in
  `CITATION.cff`. The authors must make this legal choice.
- [ ] Add the public repository URL to `CITATION.cff`.
- [ ] Add the final paper DOI/arXiv URL and BibTeX entry once public.
- [ ] Run the smoke tests and at least one full five-seed reproduction on a
  clean machine.
- [ ] Confirm the reported table uses validation-selected checkpoints and the
  documented split for every dataset.
- [ ] Review third-party origins and preserve any upstream notices required by
  their licenses.
- [ ] Before pushing, run `git status --ignored` and the no-data audit shown in
  `README.md`.
