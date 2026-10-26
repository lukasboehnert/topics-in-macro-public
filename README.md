# Topics in Macroeconomics (PhD)

Code and slides for the PhD course Topics in Macroeconomics, 2026/27, 2nd trimester. Instructor: Lukas Boehnert.

Course outline and assessment: [syllabus.pdf](syllabus.pdf). New material is added here week by week.

## Contents

| Week | Topic | Slides | Code |
| --- | --- | --- | --- |
| 1 | Local projections vs VARs | [week1.pdf](slides/week1.pdf) | [week1/](week1/) |

Setting up Python: [python_setup.pdf](slides/python_setup.pdf).

## Setup

The code is in Python 3.13 and uses [uv](https://docs.astral.sh/uv/) to install packages. Once, after installing uv:

```
git clone https://github.com/<user>/topics-in-macro.git
cd topics-in-macro
uv sync
```

This creates a `.venv` with everything the scripts need. To get new material each week:

```
git pull
uv sync
```

## Data

Most data are not in this repository: each week's README says how to get them, usually by running a `fetch_data_*.py` script in the week's `data/` folder. Data files you download into `data/` are ignored by git, so `git pull` never overwrites them.
