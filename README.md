# leap_XL

This is the repository for ICRA 2024 RGMC. This repository is maintained by XL team from IRM Lab, the Department of Automation, Tsinghua University.

## Quick start

Create a ROS workspace, cd into `src/`, then clone this repository.

For workspace preparation before hardware running, please see `doc/hardware_setup.md`

For common launch steps, please see `doc/launch.md`

## Contributing

### Code Formatting

This repository use [ruff](https://github.com/astral-sh/ruff) to lint and format python codes, [vscode-xml](https://github.com/redhat-developer/vscode-xml) and [vscode-yaml](https://github.com/redhat-developer/vscode-yaml) for formatting their corresponding files.
Please install all the recommended extensions in vscode (check `.vscode/extensions.json` if there's no automatic hint).

Plugins will automatically format the code when saving a file.

### Git Workflow

Please try to use [github-flow](https://docs.github.com/en/get-started/using-github/github-flow) for git workflow. Fork a feature branch from main branch, develop on the feature branch, and create a pull request when the feature is done.

**Create pull requests** instead of commit directly on main branch.

If main branch is updated while you are developing on a feature branch, try to use `git pull --rebase` to avoid merging commits in feature branch.

When merging pull requests, please use [squash-merge](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges) to keep a linear and simple commit history on main branch.
