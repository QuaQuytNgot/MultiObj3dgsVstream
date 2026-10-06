# Publishing the project source

This project's remote is
`https://github.com/QuaQuytNgot/MultiObj3dgsVstream.git`. Run Git commands from the
`MultiObj3dgsVstream/` root. Preparation implementation, configs, tests and docs
belong to this repository; upstream repositories are represented by pinned Git
submodules. Push the parent repository without copying upstream source into it.

Before committing, review the actual changed files and submodule pointers:

```bash
git status --short
git diff --stat
git diff --check
git submodule status --recursive
```

Select the intended project files with `git add`, then review the staged result:

```bash
git diff --cached --stat
git diff --cached --check
git diff --cached
```

`output/`, datasets, checkpoints, vendor/CUDA binaries, caches and generated media
stay on separate storage. Do not include environment directories or credentials.
Historical JSON summaries in `docs/validation/` are small provenance artifacts;
their measured paths/hashes remain unchanged and do not imply assets are bundled.

Once the staged result is reviewed, normal `git commit` and `git push` publish the
source. This migration prepares files for that review; it does not itself commit,
push or upload a backup. `scripts/github_backup_files.txt` is a project-owned file
inventory, not permission to publish a separate upstream renderer tree.
