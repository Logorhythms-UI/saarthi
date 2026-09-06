# Saarthi

Edge Efficient Episodic Memory Aid

## Local Project Setup

1. Clone the repo
1. Make sure you have [uv](https://docs.astral.sh/uv/) installed.
1. Run: `uv sync` in the project directory.
1. Since you already have `uv` installed, install `pre-commit` with `uv tool install pre-commit`.
1. Then run `pre-commit install` in this project directory.
1. Update the `pre-commit` to use the latest versions of `ruff` and `black` with `pre-commit autoupdate` in this project directory.
1. That's it! You're good to begin development :)

## Mobile Image Upload

Start Saarthi with `python main.py`, then open `http://<computer-ip>:8080/mobile-upload` on a phone connected to the same Wi-Fi network. The mobile page accepts camera captures and gallery selections. One image uses `POST /api/v1/images`; multiple images are packaged in the browser as a ZIP and use `POST /api/v1/images/zip`.

## A few ground rules

1. This project is divided into modules, and each module has its own readme stating its purpose.
2. This project will primarily be developed in python with uv as the project management tool.
3. Code quality is of prime importance in this project. Therefore, I urge the contributors to not rely blindly on LLMs.
4. Any change must be submitted as a PR first, and it must pass all code quality checks.

## Code Standards

- All the contributors must first go through [Tiger Style of Coding](https://tigerstyle.dev/).
- Though written for Zig, the Tiger Style has some incredible principles that can be incorporated into any project irrespective of language.
- In short if you are contributing to this project, you must confirm to the below standards:
  - Always mention data types next to variables, function formal arguments, and return types in function definition; do not keep the reader guessing
  - Always assert function parameters
  - Always handle exceptions explicitly
  - Use simple control flow, and avoid recursion unless really needed
  - Ensure to log things often
  - Do not use wildcard imports
  - Comments must have defined purpose and meaning in a given context, do not comment what is already obvious
  - Use a code formatter (preferably [Black](https://pypi.org/project/black/)) and always format your code before committing
  - Use `snake_case` naming convention
  - Write descriptive commit messages. PRs don't survive `git blame` commits are what future contributors are going to read
