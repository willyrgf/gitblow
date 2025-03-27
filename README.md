# GitBlow

GitBlow is a simply script that analyzes your Git repository history and present its progression.

The tool will analyze all commits in the repository, log it in stdout and display four graphs:
1. Cumulative Line Changes over time
2. Line Changes per commit
3. Cumulative Size Changes over time
4. Net Changes per commit (lines and MB)


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
pip install matplotlib gitpython numpy
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