# Environment Setup Protocol
## 3DCV Tutorials — Ultralytics + CUDA GPU Environment
**Author:** Eden Seman
**For:** Students without access to the original account
**OS:** Windows 10/11 (AMD64)

---

## Overview

This protocol walks you through setting up a fully GPU-accelerated Python environment for running the 3DCV tutorial notebooks. You will:

1. Install Git
2. Clone the required repositories
3. Install `uv` (fast Python package manager)
4. Create a virtual environment
5. Install PyTorch with CUDA support
6. Install Ultralytics and other dependencies
7. Register the environment as a Jupyter kernel in VS Code

Estimated time: **15–25 minutes**

---

## Prerequisites

- A Windows machine with an **NVIDIA GPU**
- `nvidia-smi` working in your terminal (verify your driver is installed)
- VS Code installed ([https://code.visualstudio.com](https://code.visualstudio.com))
- The **Jupyter extension** installed in VS Code

---

## Step 1 — Install Git

1. Go to [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Download and run the installer
3. Accept all defaults during installation
4. Open a **new terminal** (Command Prompt or PowerShell) and verify:

```bash
git --version
```

You should see something like `git version 2.x.x`

---

## Step 2 — Clone the Required Repositories

Navigate to your Documents folder and clone both repositories.

```bash
cd C:\Users\<your-username>\Documents
```

> **Eden's path for reference:**
> `C:\Users\edseman\Documents`
>
> **Your path will look like:**
> `C:\Users\<your-username>\Documents`

Now clone the two repos:

```bash
git clone https://github.com/ultralytics/ultralytics.git
git clone https://github.com/your-lab/3dcv_tutorials.git
```

> ⚠️ Replace the `3dcv_tutorials` URL with the actual repository URL provided by your instructor.

After cloning, your Documents folder should contain:

```
Documents\
├── ultralytics\
└── 3dcv_tutorials\
```

---

## Step 3 — Install `uv` (Python Package Manager)

`uv` is a fast, modern Python package manager. Install it via PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Add `uv` to your PATH permanently

After installation, the installer will tell you to add a path. Instead of using the temporary `set` command, run this in PowerShell to make it permanent:

```powershell
[Environment]::SetEnvironmentVariable("Path", "C:\Users\<your-username>\.local\bin;" + [Environment]::GetEnvironmentVariable("Path", "User"), "User")
```

> **Eden's version for reference:**
> `C:\Users\edseman\.local\bin`
>
> **Your version:**
> `C:\Users\<your-username>\.local\bin`

**Close and reopen your terminal**, then verify:

```bash
uv --version
```

---

## Step 4 — Create a `uv` Virtual Environment

Navigate into the cloned ultralytics folder and create the virtual environment there:

```bash
cd C:\Users\<your-username>\Documents\ultralytics
uv venv .venv
```

Now activate it:

```bash
.venv\Scripts\activate
```

✅ You will see `(.venv)` at the start of your command prompt — this means the environment is active. This is expected and correct.

---

## Step 5 — Install PyTorch with CUDA Support

> ⚠️ Do **not** skip this step and go straight to installing ultralytics — the default install will grab a CPU-only version of PyTorch and your GPU will not be detected.

With your venv active, install PyTorch built for CUDA 12.6 (compatible with CUDA 13.x drivers):

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### Verify GPU is detected

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

✅ Expected output:
```
2.x.x+cu130
CUDA: True
NVIDIA GeForce RTX XXXX   ← your GPU name here
```

If `CUDA: False`, stop here and ask your instructor before continuing.

---

## Step 6 — Install Ultralytics and Dependencies

With your venv still active, install ultralytics:

```bash
uv pip install -U ultralytics
```

Then install any additional packages required by the tutorials:

```bash
uv pip install sleap-io opencv-python numpy matplotlib
```

> Install any other missing packages the same way: `uv pip install <package-name>`
> If a notebook cell throws `ModuleNotFoundError: No module named 'xyz'`, just run `uv pip install xyz`

---

## Step 7 — Register the Environment as a Jupyter Kernel

So that VS Code can find and use this environment for notebooks, install `ipykernel` and register it:

```bash
uv pip install ipykernel
python -m ipykernel install --user --name ultralytics-env --display-name "Python (ultralytics)"
```

Verify the kernel was registered successfully:

```bash
jupyter kernelspec list
```

You should see `ultralytics-env` in the output list.

---

## Step 8 — Open the Tutorials in VS Code

Open your 3DCV tutorials folder in VS Code:

```bash
cd C:\Users\<your-username>\Documents\3dcv_tutorials
code .
```

---

## Step 9 — Select the Kernel in VS Code

1. Open any `.ipynb` notebook file in VS Code
2. In the **top-right corner**, click the kernel selector (it may say "Select Kernel" or show a Python version)
3. Choose **"Python (ultralytics)"** from the dropdown

> If the kernel doesn't appear, press `Ctrl+Shift+P` → type **"Developer: Reload Window"** → press Enter, then try again.

---

## ✅ You're All Set

Your environment is now configured with:

| Component | Version/Details |
|-----------|----------------|
| Package manager | `uv` |
| PyTorch | cu130 (CUDA-enabled) |
| Ultralytics | latest |
| Kernel | `Python (ultralytics)` in VS Code |

You can now run all notebooks in the `3dcv_tutorials` folder using the GPU-accelerated ultralytics environment.

---

## Troubleshooting

| Problem | Fix |
|--------|-----|
| `uv` not recognized | Reopen terminal after adding to PATH |
| `CUDA: False` | Make sure you installed torch with `--index-url https://download.pytorch.org/whl/cu130` and not the default |
| Kernel not showing in VS Code | Run `jupyter kernelspec list` to confirm registration, then reload VS Code window |
| `ModuleNotFoundError` in notebook | Run `uv pip install <module-name>` with venv active, then restart the kernel |
| `.venv\Scripts\activate` fails | Make sure you are inside `Documents\ultralytics` before running it |

---

*Protocol written by Eden Seman. For issues, contact your lab instructor.*
