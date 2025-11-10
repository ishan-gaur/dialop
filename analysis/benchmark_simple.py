import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load human data
print("Loading human logs...")
human_logs = []
with open(Path("/Users/ishangaur/projects/dialop/dialop/data/optimization.jsonl")) as f:
    for line in f:
        human_logs.append(json.loads(line.strip()))

print(f"Loaded {len(human_logs)} human logs")

# Extract scores
human_scores = [log["result"]["norm"] for log in human_logs]
print(f"Extracted {len(human_scores)} human scores")

# Calculate word counts
print("Calculating human word counts...")
human_word_counts = []
for log in human_logs:
    wc = 0
    for action in log["action_log"]:
        if "message" in action and "data" in action["message"]:
            wc += len(action["message"]["data"].replace("\n", " ").split(" "))
    human_word_counts.append(wc)

print(f"Calculated {len(human_word_counts)} human word counts")

# Create figure with marginal distributions
print("Creating figure...")
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                       hspace=0.1, wspace=0.1)

# Main scatter plot
ax_main = fig.add_subplot(gs[1, 0])
ax_main.scatter(human_scores, human_word_counts, label="human", alpha=0.7, color='blue', s=30)
ax_main.set_xlabel("Score Normalized")
ax_main.set_ylabel("Word Count")
ax_main.legend()
ax_main.grid(True, alpha=0.3)

# Top marginal (x-axis distribution - scores)
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
ax_top.hist(human_scores, bins=20, alpha=0.7, density=True, color='blue', edgecolor='black', linewidth=0.5)
ax_top.set_ylabel("Density")
ax_top.grid(True, alpha=0.3)
plt.setp(ax_top.get_xticklabels(), visible=False)

# Right marginal (y-axis distribution - word counts)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
ax_right.hist(human_word_counts, bins=20, alpha=0.7, density=True,
              orientation='horizontal', color='blue', edgecolor='black', linewidth=0.5)
ax_right.set_xlabel("Density")
ax_right.grid(True, alpha=0.3)
plt.setp(ax_right.get_yticklabels(), visible=False)

# Add some statistics as text
mean_score = sum(human_scores) / len(human_scores)
mean_wc = sum(human_word_counts) / len(human_word_counts)
ax_main.text(0.05, 0.95, f'Mean Score: {mean_score:.3f}\nMean Word Count: {mean_wc:.0f}',
             transform=ax_main.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle("Human Performance: Score vs Word Count with Marginal Distributions", fontsize=14)
plt.savefig('human_score_vs_wordcount_marginals.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'human_score_vs_wordcount_marginals.png'")

# Print some basic statistics
print(f"\nStatistics:")
print(f"Number of games: {len(human_logs)}")
print(f"Score range: {min(human_scores):.3f} - {max(human_scores):.3f}")
print(f"Word count range: {min(human_word_counts)} - {max(human_word_counts)}")
print(f"Mean score: {mean_score:.3f}")
print(f"Mean word count: {mean_wc:.0f}")
