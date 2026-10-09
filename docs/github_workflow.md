# GitHub workflow

Create empty private repository called `RATO`, configure SSH authentication, then:

```bash
git init -b main
git add .
git commit -m "Initial independent RATO reproduction"
git remote add origin git@github.com:YOUR_USERNAME/RATO.git
git push -u origin main
```

For changes:

```bash
git switch -c feature/update-noma
# edit and test
python -m pytest -q
git add src/rato/communication/noma.py tests/test_noma.py
git commit -m "Validate NOMA SIC model"
git push -u origin feature/update-noma
```

Open a pull request and merge after testing. Do not commit the copyrighted paper, model weights, data files or secrets.

Alternatively, from inside the extracted RATO folder use `bash scripts/init_git.sh YOUR_USERNAME`. It initializes and commits the local project and configures the remote, but intentionally does not push until you confirm the remote exists.
