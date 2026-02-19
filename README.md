# LearningAI

# Learn ML

---

## ⚙️ Setup Instructions

This project uses **Poetry** for dependency management.

### 1. Install Poetry

```bash
pip install poetry
```

> On **Windows**, ensure Python is installed and added to PATH.
> Restart PowerShell after installation.

### 2. Initialize Poetry Project

If starting fresh, initialize the project:

```bash
poetry init
```

Follow the interactive prompts to set up your `pyproject.toml` file.

### 3. Configure Python Version

Set the project to use Python 3.10 (Example):

```bash
poetry env use python3.10
```

> On **Windows**:
> * Use `py -3.10` if `python3.10` is not available
> * Verify with `py --list`

### 4. Verify Environment

Check that the correct Python version is being used:

```bash
poetry env info
```

### 5. Install Dependencies

Install all project dependencies defined in `pyproject.toml`:

```bash
poetry install
```

### 6. Adding Dependencies

To add new packages to your project:

```bash
poetry add <package-name>
```

For development dependencies:

```bash
poetry add --group dev <package-name>
```

### 7. Exporting Dependencies

To export your dependencies to `requirements.txt` (if needed for compatibility):

```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

### 8. Run Python File (Example)

```bash
python scripts/model_predict.py
```

### 9. Notes

> See also: [Poetry vs PyCharm Virtual Environment Handling](#poetry-vs-pycharm-virtual-environment-handling)
> and [Refresh Environment](#refresh-environment)

---

<!-- NOTE: Below line is linked by internal anchors. -->

## ⚙️ Poetry vs PyCharm Virtual Environment Handling

Poetry normally creates virtual environments in a central internal location, not inside your project folder.
PyCharm, however, expects a project-local .venv interpreter unless you manually point it to Poetry’s environment.

If you create an environment through PyCharm and another through Poetry, they will be two different environments,
leading to mismatched packages and scripts failing to run.

To avoid this, keep PyCharm and Poetry using the same .venv by enabling Poetry’s in-project virtualenvs or by manually
configuring PyCharm to use Poetry’s environment.

To keep both tools in sync, configure Poetry to always create the environment inside the project:

```
poetry config virtualenvs.in-project true
```

This ensures Poetry creates .venv in your project root, allowing PyCharm to automatically detect and use the same
interpreter.

To verify if Poetry creates the virtual environment inside your project, run:

```
poetry config virtualenvs.in-project
```

* If it returns true, your project is already using .venv correctly.

* If it returns false or nothing, enable in-project environments

You can also confirm by checking the active environment path:

```
poetry env info --path
```

If the path includes your project folder’s .venv, everything is correctly configured.

> Refer to the [Hard refresh](#2-hard-refresh) for steps that may resolve issues between Poetry and PyCharm
> environments


---

<!-- NOTE: Below line is linked by internal anchors. -->

## 🔄 Refresh Environment

### 1. Light refresh

When you change pyproject.toml, Poetry does NOT automatically reinstall anything. To refresh the environment cleanly,
you have to reset depending on how deep you want the cleanup to be.

**Run:**

```
poetry lock
poetry install
```

**This will:**

* Recalculate versions
* Update poetry.lock
* Install/uninstall changed dependencies

This is enough if your environment isn’t corrupted.

<!-- NOTE: Below line is linked by internal anchors. -->

### 2. Hard refresh

Use this when your Poetry environment becomes corrupted, PyCharm is using the wrong interpreter, or dependencies are not
resolving correctly.

This process completely resets the virtual environment and installs everything from scratch.

#### Step 1 — Deactivate any active venv

If you see something like (glaukos-tm-py3.10) in your terminal, deactivate it:

```
deactivate
```

#### Step 2 — Remove the existing virtual environment

Poetry may not always remove environments by Python version name, so the safest option is to delete the .venv folder
manually:

```
rm -rf .venv
```

#### Step 3 — Reinstall dependencies from scratch

Poetry will automatically recreate the virtual environment using the correct Python version (based on your
pyproject.toml constraints):

```
poetry install
```

#### Step 4 — Activate the new Poetry environment

Poetry 2.0 removed the poetry shell command unless you install a plugin.
Use the recommended new command:

```
poetry env activate
```

This prints a command like:

```
source /path/to/project/.venv/bin/activate
```

Copy and run that command:

```
source /path/to/project/.venv/bin/activate
```

#### Step 5 — Verify environment & packages

Test that Python and all required libs are correctly installed:

```
python -c "import torch, numpy, pandas; print(torch.__version__, numpy.__version__, pandas.__version__)"
```

Expected output should match the versions Poetry installed, e.g.:

```
2.9.1 1.26.4 2.3.3
```

You can also test a single package:

```
python -c "import numpy as np; print(np.__version__)"
```

**You now have:**

* A clean .venv
* Correct Python version
* Packages installed correctly
* Poetry + PyCharm back in sync

---

# ModuleNotFoundError: No module named `pkg_resources`

## Description

While attempting to use TensorFlow Hub (`tensorflow_hub`), the following error occurred:

```
ModuleNotFoundError: No module named 'pkg_resources'
```

This happened during the import step:

```python
import tensorflow_hub as hub
```

> The issue is caused by newer versions of `setuptools` (>=81) where `pkg_resources` is no longer bundled by default. Since `tensorflow_hub` internally depends on `pkg_resources`, the import fails in environments using newer setuptools versions (especially with Python 3.12).

---

## Fix

Two possible solutions:

### ✅ Option 1: Install setuptools (if missing)

```bash
poetry add setuptools
```

### ✅ Option 2: Downgrade setuptools *(recommended — what worked in this case)*

Downgrading to a version that still includes `pkg_resources` resolved the issue:

```bash
poetry add setuptools@80.0.0
```

---

## ⚙️ IDE Configurations

### 1. PyCharm

**Disable AI Inline Auto-Completion:**

---

## 🧰 Helpful Tools

### 1. Image Color Picker

[Image Color Picker](https://imagecolorpicker.com/) is used for segmentation mask labeling to extract exact HEX, RGB,
and HSV color values for different classes and objects.

---

