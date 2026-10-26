# Topics in Macroeconomics (PhD)

Code and slides for the PhD course Topics in Macroeconomics, 2026/27, 2nd trimester. Instructor: Lukas Boehnert.

Course outline and assessment: [syllabus.pdf](syllabus.pdf). New material is added here week by week.

## Contents

| Week | Topic | Slides | Code |
| --- | --- | --- | --- |
| | **Part I: Empirical methods** | | |
| 1 | Local projections vs VARs | [week1.pdf](slides/week1.pdf) | [week1/](week1/) |
| 2 | Shift-share instrumental variables | coming soon | [week2/](week2/) |
| | **Part II: International macroeconomics** | | |
| 3 | Financial crises and policy responses in open economies | coming soon | [week3/](week3/) |
| 4 | Monetary policy in open economies | coming soon | [week4/](week4/) |
| | **Part III: Macro-development** | | |
| 5 | Misallocation and growth | coming soon | [week5/](week5/) |
| 6 | Introduction to spatial economics | coming soon | [week6/](week6/) |

## Getting started

### 1. Install the tools (once)

You need [uv](https://docs.astral.sh/uv/) (installs Python and the packages), [VS Code](https://code.visualstudio.com) with the Python and Jupyter extensions, and git. The [setup slides](slides/python_setup.pdf) walk through uv and VS Code step by step. For git:

- **Mac:** open Terminal and run `git --version`. If git is missing, macOS offers to install it: click *Install*.
- **Windows:** install it from [git-scm.com](https://git-scm.com/downloads/win) with the default options, then restart VS Code.

### 2. Get the code (once)

Clone this repository into a folder **outside** OneDrive, iCloud or Dropbox (for example `Documents/phd/`). Cloning, rather than downloading a zip, lets you get each new week with one click.

**In VS Code:** press Cmd+Shift+P (Mac) or Ctrl+Shift+P (Windows), type **Git: Clone**, and paste

```
https://github.com/lukasboehnert/topics-in-macro-public.git
```

Pick the folder to put it in, and click **Open** when VS Code asks.

**Or in a terminal:**

```
cd Documents/phd
git clone https://github.com/lukasboehnert/topics-in-macro-public.git
code topics-in-macro-public
```

### 3. Install the packages

In the VS Code terminal (View → Terminal), in the `topics-in-macro-public` folder:

```
uv sync
```

This creates a `.venv` folder with Python 3.13 and the exact package versions the course uses. When VS Code asks which Python interpreter to use, pick the one in `.venv`.

### 4. Run the code

Each week's folder has a README that says what to run. For example:

```
cd week1
uv run week1_simple.py
```

Or open the script in VS Code and run it cell by cell with Shift+Enter.

## Each week: get the new material

In VS Code, click the **Sync** button (the circular arrows) at the bottom left of the window, or go to the Source Control panel and choose **⋯ → Pull**. In a terminal:

```
git pull
uv sync
```

`uv sync` is only needed when a week adds new packages, but it does no harm to run it every time.

## Where to put your own work

`git pull` updates the course files. To keep your own changes safe, **don't edit the course files directly**. Save a copy under a new name in the same folder (for example `exercise_PT_student.py` → `my_exercise_PT.py`) and work in that copy. Files you create are never touched by `git pull`, and keeping them in the same folder means the paths to `data/` still work.

If you have already edited a course file and `git pull` refuses to run ("Your local changes would be overwritten"), save your version under a new name, then restore the original and pull:

```
git restore week1/exercise_PT_student.py
git pull
```

## Data

The data for each week are in its `data/` folder, together with the `fetch_data_*.py` script that downloaded them. You don't need to run these scripts. If you do, to get newer data, the course files change: save the new data under a different name first, or restore the original with `git restore week1/data` before the next `git pull`.

## No git?

You can also click the green **Code** button at the top of this page and choose **Download ZIP**. You then have to download the zip again each week, and unpacking it over your old copy overwrites any course files you changed.
