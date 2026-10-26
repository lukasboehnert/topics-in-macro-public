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

Setting up Python: [python_setup.pdf](slides/python_setup.pdf).

## Setup

The code is in Python 3.13 and uses [uv](https://docs.astral.sh/uv/) to install packages. Once, after installing uv:

```
git clone https://github.com/lukasboehnert/topics-in-macro-public.git
cd topics-in-macro-public
uv sync
```

This creates a `.venv` with everything the scripts need. To get new material each week:

```
git pull
uv sync
```

## Data

Most data are not in this repository: each week's README says how to get them, usually by running a `fetch_data_*.py` script in the week's `data/` folder. Data files you download into `data/` are ignored by git, so `git pull` never overwrites them.
