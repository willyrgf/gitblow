import matplotlib.pyplot as plt
from datetime import datetime
import sys
import git
import os
import numpy as np

def main():
    try:
        # Try to get the git repository in the current directory
        try:
            repo = git.Repo(os.getcwd())
            if repo.bare:
                print("Error: Repository is bare")
                return 1
        except git.exc.InvalidGitRepositoryError:
            print("Error: Not a valid git repository")
            return 1
        
        # Get all commits in chronological order
        commits = list(repo.iter_commits(reverse=True))
        if not commits:
            print("No commits found in this repository")
            return 1
        
        print(f"Found {len(commits)} commits")
        
        # Initialize lists to store data
        timestamps = []
        lines_added_list = []
        lines_removed_list = []
        mb_added_list = []
        mb_removed_list = []
        
        # Process each commit
        for i, commit in enumerate(commits):
            try:
                timestamps.append(commit.committed_date)
                
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
                    diffs = parent.diff(commit, create_patch=True)
                else:
                    diffs = commit.diff(git.NULL_TREE, create_patch=True)
                
                for diff in diffs:
                    # Handle line changes
                    if hasattr(diff, 'a_blob') and diff.a_blob and hasattr(diff, 'b_blob') and diff.b_blob:
                        try:
                            # Count lines in the patch
                            patch = diff.diff.decode('utf-8', errors='replace')
                            for line in patch.split('\n'):
                                if line.startswith('+') and not line.startswith('+++'):
                                    lines_added += 1
                                elif line.startswith('-') and not line.startswith('---'):
                                    lines_removed += 1
                            
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
                            # Count lines in the file
                            content = diff.b_blob.data_stream.read().decode('utf-8', errors='replace')
                            lines_added += len(content.split('\n'))
                            
                            # Size added
                            size_added += diff.b_blob.size
                        except Exception as e:
                            print(f"Error processing new file: {e}")
                    # Handle deleted files (only a_blob exists)
                    elif hasattr(diff, 'a_blob') and diff.a_blob:
                        try:
                            # Count lines in the file
                            content = diff.a_blob.data_stream.read().decode('utf-8', errors='replace')
                            lines_removed += len(content.split('\n'))
                            
                            # Size removed
                            size_removed += diff.a_blob.size
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
        
        # Check if we have data to plot
        if not timestamps:
            print("No data to plot")
            return 1
            
        print(f"Data points: {len(timestamps)}")
        print(f"Lines added: {sum(lines_added_list)}, Lines removed: {sum(lines_removed_list)}")
        print(f"MB added: {sum(mb_added_list):.2f}, MB removed: {sum(mb_removed_list):.2f}")
        
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
        
        # Create the plots - 2x2 grid
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Cumulative lines added/removed over time
        ax1.plot(dates, cumulative_lines_added, label='Cumulative Lines Added', color='darkgreen')
        ax1.plot(dates, cumulative_lines_removed, label='Cumulative Lines Removed', color='darkred')
        ax1.fill_between(dates, cumulative_lines_added, alpha=0.3, color='green')
        ax1.fill_between(dates, cumulative_lines_removed, alpha=0.3, color='red')
        ax1.set_ylabel('Cumulative Lines')
        ax1.set_title('Cumulative Line Changes')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Lines added/removed per commit
        ax2.bar([d.strftime('%Y-%m-%d %H:%M') for d in dates], lines_added_list, label='Lines Added', color='green', alpha=0.7)
        ax2.bar([d.strftime('%Y-%m-%d %H:%M') for d in dates], [-r for r in lines_removed_list], label='Lines Removed', color='red', alpha=0.7)
        ax2.set_ylabel('Lines per Commit')
        ax2.set_title('Line Changes per Commit')
        ax2.legend()
        ax2.grid(True)
        
        # If we have multiple commits, make the x-axis more readable
        if len(dates) > 5:
            ax2.set_xticklabels([])
        elif len(dates) > 1:
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
        # Plot 3: Cumulative MB added/removed over time
        ax3.plot(dates, cumulative_mb_added, label='Cumulative MB Added', color='darkblue')
        ax3.plot(dates, cumulative_mb_removed, label='Cumulative MB Removed', color='darkorange')
        ax3.fill_between(dates, cumulative_mb_added, alpha=0.3, color='blue')
        ax3.fill_between(dates, cumulative_mb_removed, alpha=0.3, color='orange')
        ax3.set_ylabel('Cumulative MB')
        ax3.set_title('Cumulative Size Changes')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Net change per commit (added - removed)
        bars4 = ax4.bar([d.strftime('%Y-%m-%d %H:%M') for d in dates], net_lines, label='Net Lines', color=['green' if x >= 0 else 'red' for x in net_lines], alpha=0.7)
        ax4_twin = ax4.twinx()
        line4 = ax4_twin.plot(range(len(dates)), net_mb, label='Net MB', color='blue', marker='o', linestyle='-')
        ax4.set_ylabel('Net Lines per Commit')
        ax4_twin.set_ylabel('Net MB per Commit')
        ax4.set_title('Net Changes per Commit')
        
        # Combine legends for both y-axes
        lines, labels = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines + lines2, labels + labels2, loc='upper left')
        ax4.grid(True)
        
        # If we have multiple commits, make the x-axis more readable
        if len(dates) > 5:
            ax4.set_xticklabels([])
        elif len(dates) > 1:
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
        # Adjust layout
        plt.tight_layout()
        
        # Format x-axis date labels
        if len(dates) > 1:
            fig.autofmt_xdate()
        
        # Add a title
        plt.suptitle('Git Repository Changes Analysis', fontsize=16, y=0.98)
        
        # Display the plot
        plt.show()
        
        return 0
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())