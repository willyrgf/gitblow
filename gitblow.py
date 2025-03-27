import matplotlib.pyplot as plt
from datetime import datetime
import sys
import git
import os

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
        
        # Convert timestamps to datetime objects for plotting
        dates = [datetime.fromtimestamp(t) for t in timestamps]
        
        # Create the plots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        
        # Plot 1: Lines added and removed
        ax1.plot(dates, lines_added_list, label='Lines Added', color='green', marker='o')
        ax1.plot(dates, lines_removed_list, label='Lines Removed', color='red', marker='x')
        ax1.set_ylabel('Lines')
        ax1.set_title('Line Changes Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Add some padding to make sure all points are visible
        if len(dates) > 1:
            ax1.margins(0.05)
        
        # Plot 2: MB added and removed
        ax2.plot(dates, mb_added_list, label='MB Added', color='blue', marker='o')
        ax2.plot(dates, mb_removed_list, label='MB Removed', color='orange', marker='x')
        ax2.set_ylabel('MB')
        ax2.set_xlabel('Time')
        ax2.set_title('Size Changes Over Time')
        ax2.legend()
        ax2.grid(True)
        
        # Add some padding to make sure all points are visible
        if len(dates) > 1:
            ax2.margins(0.05)
            
        # Adjust layout
        plt.tight_layout()
        
        # Format x-axis date labels if we have multiple data points
        if len(dates) > 1:
            fig.autofmt_xdate()
        
        # Add a title
        plt.suptitle('Git Repository Changes Over Time', fontsize=16, y=1.02)
        
        # Display the plot
        plt.show()
        
        return 0
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())