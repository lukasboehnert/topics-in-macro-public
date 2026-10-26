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

Data are not in this repository. Each week's README says how to get them: most are downloaded by a `fetch_data_*.py` script in the week's `data/` folder, and the rest are on the course page. Data files you put in `data/` are ignored by git, so `git pull` never overwrites them.
