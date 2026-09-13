# Contributing

Issues and pull requests are welcome. Please keep competition data, labels,
predictions, checkpoints, access tokens, and private server details out of the
repository. New result claims should include the split, metric implementation,
selection boundary, random seed, and an aggregate machine-readable artifact.

Run the lightweight checks before opening a pull request:

```bash
python -m pytest -q
python -m compileall -q src
```
