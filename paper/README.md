# Paper

`main.tex` builds in plain `article` class so it compiles anywhere:

```bash
make            # regenerates dataset_stats.tex + results_table.tex, then builds main.pdf
```

`dataset_stats.tex` and `results_table.tex` are **auto-generated** from the real
data and pilot results by `experiments/make_paper_assets.py` — do not edit by hand.

## Switching to the official EMNLP/ACL template
For submission, download the ACL template (`acl.sty`, `acl_natbib.bst`, `acl.bib`
conventions) from the ACL Rolling Review / EMNLP author kit and:
1. replace `\documentclass[11pt]{article}` + the `geometry/times` lines with
   `\documentclass[11pt]{article}\usepackage[review]{acl}`;
2. change `\bibliographystyle{plainnat}` to `\bibliographystyle{acl_natbib}`.

The section structure and `\input{}` of generated assets stay the same.
