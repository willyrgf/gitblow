# GitBlow

GitBlow is a simply script that analyzes your Git repository history and present its progression.


## Installation

### Using Nix (recommended)

The easiest way to install GitBlow is using Nix:

```bash
# Run directly without installing
nix run github:willyrgf/gitblow

# Or install it to your profile
nix profile install github:willyrgf/gitblow
```

### Using Python and pip

1. Clone the repository:
```bash
git clone https://github.com/willyrgf/gitblow.git
cd gitblow
```

2. Install dependencies:
```bash
pip install bokeh gitpython numpy
```

3. Run the script:
```bash
python gitblow.py
```

## Usage

Navigate to any Git repository and run:

```bash
gitblow
```

Or analyze a specific local repository:

```bash
gitblow --repo-path /path/to/repository
# or with short option
gitblow -p /path/to/repository
```

You can also analyze a remote repository without cloning it manually:

```bash
gitblow --repo-url https://github.com/user/repo.git
# or with short option
gitblow -u https://github.com/user/repo.git
```

For large repositories, you can limit the number of commits to analyze:

```bash
gitblow --max-commits 100
# or with short option
gitblow -m 100
```

## Visualization Features

GitBlow provides an interactive HTML visualization that adapts based on the size of your repository:

1. **Adaptive Layout**: For repositories with many commits (>50), GitBlow automatically switches to a scatter plot visualization with connecting stems to prevent visual clutter.

2. **Time Range Selection**: All visualizations include a date range slider that allows you to focus on specific time periods in your repository history.

3. **Interactive Tooltips**: Hover over any data point to see detailed information about the commit, including hash, author, message, and exact numbers.

4. **Repository-named Output**: Visualization files are automatically named with the repository name and timestamp for easy identification.

## License

This project is licensed under the MIT License - see the LICENSE file for details.