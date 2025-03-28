#!/usr/bin/env python3
import sys
import git
import os
import shutil
import argparse
import tempfile
import numpy as np
import time
import pickle
import hashlib
from datetime import datetime, timedelta
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column, row, layout
from bokeh.models import ColumnDataSource, HoverTool, Range1d, Slider, DateRangeSlider, CustomJS, Panel, Tabs, CheckboxGroup, LinearAxis, Span, Label, Text
from bokeh.palettes import Viridis256, Category10
from bokeh.io import reset_output

# Global variable for command line arguments
args = None

def analyze_repository(repo_path, max_commits=None, use_cache=True):
    """Analyze a git repository and create visualization."""
    global args
    try:
        start_time = time.time()
        
        # Try to get the git repository
        try:
            repo = git.Repo(repo_path)
            if repo.bare:
                print("Error: Repository is bare")
                return 1
        except git.exc.InvalidGitRepositoryError:
            print(f"Error: Not a valid git repository at {repo_path}")
            return 1
        
        # Initialize variables that will be filled either from cache or from analysis
        timestamps = []
        lines_added_list = []
        lines_removed_list = []
        mb_added_list = []
        mb_removed_list = []
        commit_msgs = []
        commit_hashes = []
        authors = []
        num_commits = 0  # Initialize counter for number of commits
        
        # Create cache directory if needed
        cache_dir = None
        cached_data_loaded = False
        if use_cache:
            cache_dir = os.path.join(tempfile.gettempdir(), "gitblow_cache")
            os.makedirs(cache_dir, exist_ok=True)
            repo_hash = repo.git.rev_parse('HEAD')[:10]  # Use first 10 chars of HEAD as repo identifier
            cache_file = os.path.join(cache_dir, f"{repo_hash}.cache")
            
            # Check if we have a cached analysis
            if os.path.exists(cache_file):
                try:
                    print(f"Found cached analysis, loading from {cache_file}")
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        # Verify it's the same repository
                        if cache_data.get('repo_hash') == repo_hash:
                            print(f"Using cached analysis data")
                            
                            # Extract data from cache
                            timestamps = cache_data['timestamps']
                            lines_added_list = cache_data['lines_added']
                            lines_removed_list = cache_data['lines_removed']
                            mb_added_list = cache_data['mb_added']
                            mb_removed_list = cache_data['mb_removed']
                            commit_msgs = cache_data['commit_msgs']
                            commit_hashes = cache_data['commit_hashes']
                            authors = cache_data['authors']
                            
                            # Get the number of commits from the cache
                            num_commits = len(timestamps)
                            
                            # Skip to visualization
                            print(f"Data points: {len(timestamps)}")
                            print(f"Lines added: {sum(lines_added_list)}, Lines removed: {sum(lines_removed_list)}")
                            print(f"MB added: {sum(mb_added_list):.2f}, MB removed: {sum(mb_removed_list):.2f}")
                            
                            # Calculate time saved
                            elapsed = time.time() - start_time
                            print(f"Loaded from cache in {elapsed:.2f} seconds")
                            
                            # Set flag to skip analysis
                            cached_data_loaded = True
                            
                except Exception as e:
                    print(f"Error loading cache: {e}, will perform full analysis")
        
        # Only perform analysis if we didn't load from cache
        if not cached_data_loaded:
            # Get all commits in chronological order
            commits = list(repo.iter_commits(reverse=True))
            if not commits:
                print("No commits found in this repository")
                return 1
            
            total_commits = len(commits)
            print(f"Found {total_commits} commits")
            
            # Apply commit limit if specified
            if max_commits and max_commits > 0 and max_commits < total_commits:
                print(f"Limiting analysis to {max_commits} commits (out of {total_commits})")
                # Get a representative sample by taking commits from throughout the history
                if max_commits == 1:
                    commits = [commits[0]]  # Just take the first commit
                else:
                    step = total_commits / max_commits
                    if step < 1:
                        step = 1
                    indices = [int(i * step) for i in range(max_commits)]
                    if indices[-1] != total_commits - 1:  # Make sure we include the most recent commit
                        indices[-1] = total_commits - 1
                    commits = [commits[i] for i in indices]
            
            # Store the number of commits we're analyzing
            num_commits = len(commits)
            
            # Optimize diff processing with batch mode
            progress_step = max(1, len(commits) // 20)  # Show progress every 5%
            last_progress_update = time.time()
            
            # Process each commit
            for i, commit in enumerate(commits):
                try:
                    # Show progress more efficiently
                    if i % progress_step == 0 or (time.time() - last_progress_update) > 5:
                        progress = (i / len(commits)) * 100
                        elapsed = time.time() - start_time
                        eta = (elapsed / (i+1)) * (len(commits) - i - 1) if i > 0 else 0
                        print(f"Processing commit {i+1}/{len(commits)} ({progress:.1f}%) - Time elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
                        last_progress_update = time.time()
                        # Clear terminal line to update in place
                        sys.stdout.flush()
                    
                    timestamps.append(commit.committed_date)
                    commit_msgs.append(commit.message.split('\n')[0])
                    commit_hashes.append(commit.hexsha[:7])
                    authors.append(commit.author.name)
                    
                    # Get diff stats
                    if i == 0:
                        # First commit - diff with empty tree
                        parent = None
                    else:
                        parent = commits[i-1]
                    
                    # Calculate line changes
                    lines_added = 0
                    lines_removed = 0
                    size_added = 0
                    size_removed = 0
                    
                    # Get diffs between this commit and its parent
                    if parent:
                        diffs = parent.diff(commit, create_patch=True, 
                                          # Speed optimization: avoid generating too much data
                                          find_renames=False,
                                          find_copies=False)
                    else:
                        diffs = commit.diff(git.NULL_TREE, create_patch=True,
                                          find_renames=False, 
                                          find_copies=False)
                    
                    for diff in diffs:
                        # Handle line changes
                        if hasattr(diff, 'a_blob') and diff.a_blob and hasattr(diff, 'b_blob') and diff.b_blob:
                            try:
                                # Optimize: Count lines directly from diff
                                patch = diff.diff.decode('utf-8', errors='replace')
                                plus_count = 0
                                minus_count = 0
                                for line in patch.split('\n'):
                                    if line.startswith('+') and not line.startswith('+++'):
                                        plus_count += 1
                                    elif line.startswith('-') and not line.startswith('---'):
                                        minus_count += 1
                                
                                lines_added += plus_count
                                lines_removed += minus_count
                                
                                # Calculate size changes
                                if diff.a_blob and diff.b_blob:
                                    old_size = diff.a_blob.size
                                    new_size = diff.b_blob.size
                                    if new_size > old_size:
                                        size_added += (new_size - old_size)
                                    elif old_size > new_size:
                                        size_removed += (old_size - new_size)
                            except Exception as e:
                                print(f"Error processing diff: {e}")
                        # Handle new files (only b_blob exists)
                        elif hasattr(diff, 'b_blob') and diff.b_blob:
                            try:
                                # Optimization: don't decode the entire file for line counting
                                size_added += diff.b_blob.size
                                
                                # For line counts, use a more efficient approach
                                if diff.b_blob.size < 1000000:  # Only decode small files
                                    content = diff.b_blob.data_stream.read().decode('utf-8', errors='replace')
                                    lines_added += len(content.split('\n'))
                                else:
                                    # For very large files, estimate based on average line length
                                    # This avoids memory issues with massive files
                                    estimated_lines = diff.b_blob.size // 50  # Assume average 50 bytes per line
                                    lines_added += estimated_lines
                                    print(f"  Large file detected, lines estimated: {estimated_lines}")
                            except Exception as e:
                                print(f"Error processing new file: {e}")
                        # Handle deleted files (only a_blob exists)
                        elif hasattr(diff, 'a_blob') and diff.a_blob:
                            try:
                                # Same optimization for deleted files
                                size_removed += diff.a_blob.size
                                
                                if diff.a_blob.size < 1000000:
                                    content = diff.a_blob.data_stream.read().decode('utf-8', errors='replace')
                                    lines_removed += len(content.split('\n'))
                                else:
                                    estimated_lines = diff.a_blob.size // 50
                                    lines_removed += estimated_lines
                                    print(f"  Large file detected, lines estimated: {estimated_lines}")
                            except Exception as e:
                                print(f"Error processing deleted file: {e}")
                    
                    # Convert bytes to MB
                    mb_added = size_added / 1000000.0
                    mb_removed = size_removed / 1000000.0
                    
                    # Store the data
                    lines_added_list.append(lines_added)
                    lines_removed_list.append(lines_removed)
                    mb_added_list.append(mb_added)
                    mb_removed_list.append(mb_removed)
                    
                    # Print commit info
                    print(f"Commit {commit.hexsha[:7]}: +{lines_added}/-{lines_removed} lines, +{mb_added:.2f}/-{mb_removed:.2f} MB")
                    
                except Exception as e:
                    print(f"Error processing commit {commit.hexsha}: {e}")
                    continue
            
            # Save to cache if enabled
            if use_cache and cache_dir:
                try:
                    cache_data = {
                        'repo_hash': repo_hash,
                        'timestamp': time.time(),
                        'timestamps': timestamps,
                        'lines_added': lines_added_list,
                        'lines_removed': lines_removed_list,
                        'mb_added': mb_added_list,
                        'mb_removed': mb_removed_list,
                        'commit_msgs': commit_msgs,
                        'commit_hashes': commit_hashes,
                        'authors': authors
                    }
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)
                    print(f"Saved analysis to cache: {cache_file}")
                except Exception as e:
                    print(f"Error saving to cache: {e}")
            
            # Check if we have data to plot
            if not timestamps:
                print("No data to plot")
                return 1
                
            print(f"Data points: {len(timestamps)}")
            print(f"Lines added: {sum(lines_added_list)}, Lines removed: {sum(lines_removed_list)}")
            print(f"MB added: {sum(mb_added_list):.2f}, MB removed: {sum(mb_removed_list):.2f}")
            
            # Completion time
            elapsed = time.time() - start_time
            print(f"Analysis completed in {elapsed:.2f} seconds")
        
        # VISUALIZATION STARTS HERE - Either from cache or fresh analysis
        
        # Calculate cumulative changes
        cumulative_lines_added = np.cumsum(lines_added_list)
        cumulative_lines_removed = np.cumsum(lines_removed_list)
        cumulative_mb_added = np.cumsum(mb_added_list)
        cumulative_mb_removed = np.cumsum(mb_removed_list)
        
        # Calculate net diff per commit
        net_lines = [a - r for a, r in zip(lines_added_list, lines_removed_list)]
        net_mb = [a - r for a, r in zip(mb_added_list, mb_removed_list)]
        
        # Convert timestamps to datetime objects for plotting
        dates = [datetime.fromtimestamp(t) for t in timestamps]
        
        # Calculate negative values for lines removed (for the bar chart)
        neg_lines_removed = [-r for r in lines_removed_list]
        
        # Determine if we should use adaptive visualization based on commit count
        use_adaptive_viz = len(dates) > 50
        
        # Get repository name for the output file
        repo_name = "repository"
        
        # Parse repository name from URL or path
        if args and hasattr(args, 'repo_url') and args.repo_url:
            # Extract repo name from URL
            url_parts = args.repo_url.rstrip('/').split('/')
            if url_parts:
                repo_name = url_parts[-1]
                if repo_name.endswith('.git'):
                    repo_name = repo_name[:-4]
        else:
            # Extract from local path
            if '/' in repo_path:
                repo_name = os.path.basename(repo_path.rstrip('/'))
            else:
                repo_name = repo_path
            
        # Remove .git extension if present
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
            
        # Create an HTML file to display the visualization in a browser
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"gitblow_{repo_name}_{timestamp}.html"
        output_file(output_filename, title=f"Git Repository Analysis - {num_commits} commits")
        
        # Create ColumnDataSources for Bokeh
        source_cumulative = ColumnDataSource(data=dict(
            dates=dates,
            cum_lines_added=cumulative_lines_added,
            cum_lines_removed=cumulative_lines_removed,
            cum_mb_added=cumulative_mb_added,
            cum_mb_removed=cumulative_mb_removed,
            commit_hash=commit_hashes,
            commit_msg=commit_msgs,
            author=authors
        ))
        
        source_per_commit = ColumnDataSource(data=dict(
            dates=dates,
            lines_added=lines_added_list,
            lines_removed=neg_lines_removed,
            commit_hash=commit_hashes,
            commit_msg=commit_msgs,
            author=authors
        ))
        
        source_mb = ColumnDataSource(data=dict(
            dates=dates,
            mb_added=mb_added_list,
            mb_removed=mb_removed_list,
            commit_hash=commit_hashes,
            commit_msg=commit_msgs,
            author=authors
        ))
        
        colors = ['green' if x >= 0 else 'red' for x in net_lines]
        source_net = ColumnDataSource(data=dict(
            dates=dates,
            net_lines=net_lines,
            net_mb=net_mb,
            commit_hash=commit_hashes,
            commit_msg=commit_msgs,
            color=colors,
            author=authors
        ))
        
        # Enhanced tooltips for interactive hover - showing exact values
        hover_tool = HoverTool(
            tooltips=[
                ("Date", "@dates{%F %T}"),
                ("Commit", "@commit_hash"),
                ("Author", "@author"),
                ("Message", "@commit_msg"),
                ("Lines Added", "@lines_added{0,0}"),
                ("Lines Removed", "@lines_removed{0,0}"),
            ],
            formatters={"@dates": "datetime"}
        )
        
        hover_tool_net = HoverTool(
            tooltips=[
                ("Date", "@dates{%F %T}"),
                ("Commit", "@commit_hash"),
                ("Author", "@author"),
                ("Message", "@commit_msg"),
                ("Net Lines", "@net_lines{0,0}"),
                ("Net MB", "@net_mb{0.00} MB"),
            ],
            formatters={"@dates": "datetime"}
        )
        
        # Calculate time windows for large repositories
        if len(dates) > 1:
            # Create a series of appropriate time windows
            earliest_date = dates[0]
            latest_date = dates[-1]
            time_span = (latest_date - earliest_date).total_seconds()
            
            # Time periods for window selection (for large repos)
            last_month = latest_date - timedelta(days=30)
            last_quarter = latest_date - timedelta(days=90)
            last_year = latest_date - timedelta(days=365)
            
            # Initial view based on repository size
            if len(dates) > 500:
                initial_start_date = last_month
            elif len(dates) > 200:
                initial_start_date = last_quarter
            elif len(dates) > 100:
                initial_start_date = last_year
            else:
                initial_start_date = earliest_date
        else:
            earliest_date = dates[0] - timedelta(days=1)
            latest_date = dates[0] + timedelta(days=1)
            initial_start_date = earliest_date
        
        # For large repositories, determine an appropriate bar width
        if len(dates) > 1:
            # For repositories with many commits, use time-based width
            if use_adaptive_viz:
                # Calculate several options for bar width
                day_width_ms = 86400000  # 1 day in milliseconds
                if len(dates) > 500:
                    bar_width_ms = day_width_ms / 24  # 1 hour for very large repos
                elif len(dates) > 200:
                    bar_width_ms = day_width_ms / 12  # 2 hours for large repos
                elif len(dates) > 100:
                    bar_width_ms = day_width_ms / 6   # 4 hours for medium repos
                else:
                    bar_width_ms = day_width_ms / 3   # 8 hours for smaller repos
            else:
                # For smaller repos, calculate based on commit density and ensure bars don't overlap
                time_span = (latest_date - earliest_date).total_seconds() * 1000  # convert to ms
                
                # If all commits are on same day or very close, create artificial spread
                if time_span < 86400000:  # less than a day
                    time_span = max(86400000, time_span * 3)  # at least one day spread
                
                # Maximum bar width should ensure no overlap
                # Calculate width based on time span and number of commits with a spacing factor
                spacing_factor = 12  # Higher means more space between bars (increased from 8)
                calculated_width = time_span / (len(dates) * spacing_factor)
                
                # Cap the maximum width to prevent bars from being too fat
                max_width_ms = 300000  # 5 minutes max width (reduced from 15 minutes)
                min_width_ms = 30000   # 30 seconds min width (reduced from 1 minute)
                
                bar_width_ms = max(min_width_ms, min(calculated_width, max_width_ms))
        else:
            # For single commit repos, use a sensible default
            bar_width_ms = 3600000  # 1 hour
        
        # Common parameters for all plots
        TOOLS = ["box_zoom", "reset", "save", "pan", "wheel_zoom"]
        PLOT_HEIGHT = 350  # Slightly smaller to fit more on screen
        PLOT_WIDTH = 1000
        
        # Cumulative plots and per-commit plots are separated into tabs for large repos
        if use_adaptive_viz:
            # TAB 1: CUMULATIVE VIEW
            
            # Plot 1: Cumulative lines added/removed over time
            p1 = figure(title="Cumulative Line Changes", 
                      x_axis_type="datetime", 
                      x_axis_label="Time", 
                      y_axis_label="Cumulative Lines",
                      height=PLOT_HEIGHT, width=PLOT_WIDTH,
                      tools=TOOLS + [hover_tool])
            
            p1.line('dates', 'cum_lines_added', source=source_cumulative, line_width=2, 
                    line_color='darkgreen', legend_label="Cumulative Lines Added")
            p1.line('dates', 'cum_lines_removed', source=source_cumulative, line_width=2, 
                    line_color='darkred', legend_label="Cumulative Lines Removed")
            
            # Add a transparent fill
            p1.varea(x='dates', y1=0, y2='cum_lines_added', source=source_cumulative, 
                    fill_color='green', fill_alpha=0.2)
            p1.varea(x='dates', y1=0, y2='cum_lines_removed', source=source_cumulative, 
                    fill_color='red', fill_alpha=0.2)
            
            p1.legend.location = "top_left"
            p1.legend.click_policy = "hide"
            
            # Plot 3: Cumulative MB added/removed over time
            p3 = figure(title="Cumulative Size Changes", 
                      x_axis_type="datetime", 
                      x_axis_label="Time", 
                      y_axis_label="Cumulative MB",
                      height=PLOT_HEIGHT, width=PLOT_WIDTH,
                      tools=TOOLS + [hover_tool],
                      x_range=p1.x_range)  # Share the x range with p1
            
            p3.line('dates', 'cum_mb_added', source=source_cumulative, line_width=2, 
                    line_color='darkblue', legend_label="Cumulative MB Added")
            p3.line('dates', 'cum_mb_removed', source=source_cumulative, line_width=2, 
                    line_color='darkorange', legend_label="Cumulative MB Removed")
            
            # Add a transparent fill
            p3.varea(x='dates', y1=0, y2='cum_mb_added', source=source_cumulative, 
                    fill_color='blue', fill_alpha=0.2)
            p3.varea(x='dates', y1=0, y2='cum_mb_removed', source=source_cumulative, 
                    fill_color='orange', fill_alpha=0.2)
            
            p3.legend.location = "top_left"
            p3.legend.click_policy = "hide"
            
            # Create a range slider for the cumulative view
            date_range_slider1 = DateRangeSlider(
                title="Time Range (Cumulative)",
                start=earliest_date,
                end=latest_date,
                value=(initial_start_date, latest_date),
                step=24*60*60*1000,  # 1 day in ms
                width=PLOT_WIDTH
            )
            
            # Link the slider to the plots
            date_range_slider1.js_link('value', p1.x_range, 'start', attr_selector=0)
            date_range_slider1.js_link('value', p1.x_range, 'end', attr_selector=1)
            
            # Plot 2: Lines added/removed per commit
            p2 = figure(title="Line Changes per Commit", 
                      x_axis_type="datetime", 
                      x_axis_label="Time", 
                      y_axis_label="Lines per Commit",
                      height=PLOT_HEIGHT, width=PLOT_WIDTH,
                      tools=TOOLS + [hover_tool])
            
            # Add scatter points with hover for lines added
            p2.scatter('dates', 'lines_added', source=source_per_commit, 
                     size=8, color='green', alpha=0.7, legend_label="Lines Added")
            
            # Add scatter points with hover for lines removed
            p2.scatter('dates', 'lines_removed', source=source_per_commit, 
                     size=8, color='red', alpha=0.7, legend_label="Lines Removed")
            
            # Add stems to connect points to baseline (like a stem plot)
            for i, (date, value) in enumerate(zip(dates, lines_added_list)):
                if value > 0:  # Only draw stems for visible points
                    p2.line([date, date], [0, value], line_color='green', line_width=1, alpha=0.3)
            
            for i, (date, value) in enumerate(zip(dates, neg_lines_removed)):
                if value < 0:  # Only draw stems for visible points
                    p2.line([date, date], [0, value], line_color='red', line_width=1, alpha=0.3)
            
            p2.legend.location = "top_left"
            p2.legend.click_policy = "hide"
            
            # Plot 4: Net change per commit
            p4 = figure(title="Net Changes per Commit", 
                      x_axis_type="datetime", 
                      x_axis_label="Time", 
                      y_axis_label="Net Changes per Commit",
                      height=PLOT_HEIGHT, width=PLOT_WIDTH,
                      tools=TOOLS + [hover_tool_net],
                      x_range=p2.x_range)  # Share the x range with p2
            
            # Add scatter points with hover for net lines
            p4.scatter('dates', 'net_lines', source=source_net, 
                     size=8, color='color', alpha=0.7, legend_label="Net Lines")
            
            # Add stems for net lines (like a stem plot)
            for i, (date, value, color) in enumerate(zip(dates, net_lines, colors)):
                p4.line([date, date], [0, value], line_color=color, line_width=1, alpha=0.3)
            
            # Add scatter and line for MB
            p4.scatter('dates', 'net_mb', source=source_net, size=6,
                     color='blue', alpha=0.7, legend_label="Net MB")
            p4.line('dates', 'net_mb', source=source_net, color='blue', 
                   line_width=1.5, alpha=0.3)
            
            p4.legend.location = "top_left"
            p4.legend.click_policy = "hide"
            
            # Create a range slider for the per-commit view
            date_range_slider2 = DateRangeSlider(
                title="Time Range (Per-Commit)",
                start=earliest_date,
                end=latest_date,
                value=(initial_start_date, latest_date),
                step=24*60*60*1000,  # 1 day in ms
                width=PLOT_WIDTH
            )
            
            # Link the slider to the plots
            date_range_slider2.js_link('value', p2.x_range, 'start', attr_selector=0)
            date_range_slider2.js_link('value', p2.x_range, 'end', attr_selector=1)
            
            # Create a combined layout (instead of tabs)
            layout = column(
                p1, p3, date_range_slider1,
                p2, p4, date_range_slider2
            )
            
            # Show the visualization
            show(layout)
            
        else:
            # For smaller repositories, use the classic 4-panel view
            
            # Plot 1: Cumulative lines added/removed over time
            p1 = figure(title="Cumulative Line Changes", 
                       x_axis_type="datetime", 
                       x_axis_label="Time", 
                       y_axis_label="Cumulative Lines",
                       height=400, width=1000,
                       tools=TOOLS + [hover_tool])
            
            p1.line('dates', 'cum_lines_added', source=source_cumulative, line_width=2, 
                    line_color='darkgreen', legend_label="Cumulative Lines Added")
            p1.line('dates', 'cum_lines_removed', source=source_cumulative, line_width=2, 
                    line_color='darkred', legend_label="Cumulative Lines Removed")
            
            # Add a transparent fill
            p1.varea(x='dates', y1=0, y2='cum_lines_added', source=source_cumulative, 
                    fill_color='green', fill_alpha=0.2)
            p1.varea(x='dates', y1=0, y2='cum_lines_removed', source=source_cumulative, 
                    fill_color='red', fill_alpha=0.2)
            
            p1.legend.location = "top_left"
            p1.legend.click_policy = "hide"
            
            # Plot 2: Lines added/removed per commit
            p2 = figure(title="Line Changes per Commit", 
                       x_axis_type="datetime", 
                       x_axis_label="Time", 
                       y_axis_label="Lines per Commit",
                       height=400, width=1000,
                       tools=TOOLS + [hover_tool])
            
            # Add scatter points with hover for lines added/removed
            p2.scatter('dates', 'lines_added', source=source_per_commit, 
                     size=8, color='green', alpha=0.7, legend_label="Lines Added")
            p2.scatter('dates', 'lines_removed', source=source_per_commit, 
                     size=8, color='red', alpha=0.7, legend_label="Lines Removed")
            
            # Add stems to connect points to baseline
            for i, (date, value) in enumerate(zip(dates, lines_added_list)):
                if value > 0:  # Only draw stems for visible points
                    p2.line([date, date], [0, value], line_color='green', line_width=1, alpha=0.5)
            
            for i, (date, value) in enumerate(zip(dates, neg_lines_removed)):
                if value < 0:  # Only draw stems for visible points
                    p2.line([date, date], [0, value], line_color='red', line_width=1, alpha=0.5)
            
            p2.legend.location = "top_left"
            p2.legend.click_policy = "hide"
            
            # Plot 3: Cumulative MB added/removed over time
            p3 = figure(title="Cumulative Size Changes", 
                       x_axis_type="datetime", 
                       x_axis_label="Time", 
                       y_axis_label="Cumulative MB",
                       height=400, width=1000,
                       tools=TOOLS + [hover_tool])
            
            p3.line('dates', 'cum_mb_added', source=source_cumulative, line_width=2, 
                    line_color='darkblue', legend_label="Cumulative MB Added")
            p3.line('dates', 'cum_mb_removed', source=source_cumulative, line_width=2, 
                    line_color='darkorange', legend_label="Cumulative MB Removed")
            
            # Add a transparent fill
            p3.varea(x='dates', y1=0, y2='cum_mb_added', source=source_cumulative, 
                    fill_color='blue', fill_alpha=0.2)
            p3.varea(x='dates', y1=0, y2='cum_mb_removed', source=source_cumulative, 
                    fill_color='orange', fill_alpha=0.2)
            
            p3.legend.location = "top_left"
            p3.legend.click_policy = "hide"
            
            # Plot 4: Net change per commit
            p4 = figure(title="Net Changes per Commit", 
                       x_axis_type="datetime", 
                       x_axis_label="Time", 
                       y_axis_label="Net Lines per Commit",
                       height=400, width=1000,
                       tools=TOOLS + [hover_tool_net])
            
            # Add scatter and stems for net changes
            p4.scatter('dates', 'net_lines', source=source_net, 
                     size=8, color='color', alpha=0.7, legend_label="Net Lines")
            
            # Add stems for net lines
            for i, (date, value, color) in enumerate(zip(dates, net_lines, colors)):
                p4.line([date, date], [0, value], line_color=color, line_width=1, alpha=0.5)
            
            # Add MB scatter plot with smaller markers
            p4.scatter(x='dates', y='net_mb', source=source_net, size=6,
                    color='blue', alpha=0.7, legend_label="Net MB")
            p4.line('dates', 'net_mb', source=source_net, color='blue', 
                   line_width=1.5, alpha=0.5)
            
            p4.legend.location = "top_left"
            p4.legend.click_policy = "hide"
            
            # Set initial range for all plots
            date_range = Range1d(start=initial_start_date, end=latest_date + timedelta(hours=1))
            for p in [p1, p2, p3, p4]:
                p.x_range = date_range
            
            # Create date range slider for all plots
            date_range_slider = DateRangeSlider(
                title="Time Range",
                start=earliest_date,
                end=latest_date,
                value=(initial_start_date, latest_date),
                step=24*60*60*1000,  # 1 day in ms
                width=1000
            )
            
            # Link the slider to all plots
            for p in [p1, p2, p3, p4]:
                date_range_slider.js_link('value', p.x_range, 'start', attr_selector=0)
                date_range_slider.js_link('value', p.x_range, 'end', attr_selector=1)
            
            # Arrange all plots in a column with the slider at the bottom
            layout = column(p1, p2, p3, p4, date_range_slider)
            
            # Show the visualization
            show(layout)
        
        print(f"Visualization saved to {output_filename}")
        return 0
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

def main():
    """Main function to parse arguments and run the analysis."""
    global args
    parser = argparse.ArgumentParser(description='Analyze Git repository changes over time.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--repo-url', '-u', help='URL of the Git repository to analyze (http/https/ssh)')
    group.add_argument('--repo-path', '-p', help='Path to local Git repository (default: current directory)', default=os.getcwd())
    
    # Add performance-related arguments
    parser.add_argument('--max-commits', '-m', type=int, help='Maximum number of commits to analyze (default: all)', default=None)
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of analysis results')
    
    args = parser.parse_args()
    
    temp_dir = None
    try:
        if args.repo_url:
            print(f"Cloning repository from: {args.repo_url}")
            # Create temporary directory for the clone
            temp_dir = tempfile.mkdtemp(prefix="gitblow_")
            try:
                # Clone the repository
                git.Repo.clone_from(args.repo_url, temp_dir)
                print(f"Repository cloned to temporary directory: {temp_dir}")
                # Analyze the cloned repository
                return analyze_repository(temp_dir, max_commits=args.max_commits, use_cache=not args.no_cache)
            except git.exc.GitCommandError as e:
                print(f"Failed to clone repository: {e}")
                return 1
        else:
            # Analyze local repository
            return analyze_repository(args.repo_path, max_commits=args.max_commits, use_cache=not args.no_cache)
    finally:
        # Clean up temporary directory if it exists
        if temp_dir and os.path.exists(temp_dir):
            print(f"Cleaning up temporary repository: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    sys.exit(main())