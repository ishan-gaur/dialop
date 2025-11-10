import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

folders = {
    "low-effort": Path("/Users/ishangaur/projects/dialop/dialop/results/matching/test_selfplay_low"),
    "high-effort": Path("/Users/ishangaur/projects/dialop/dialop/results/matching/test_selfplay_high")
}

def extract_logs(folder):
    logs = []
    files = list(folder.glob("*.out"))
    print(f"    Found {len(files)} files to process")

    for i, file in enumerate(sorted(files)):
        if i % 10 == 0:
            print(f"    Processing file {i+1}/{len(files)}")

        try:
            text = file.read_text()
            result_st = text.find("==================== Result ====================")
            if result_st == -1:
                print(f"    Skipping {file.name} - no result section found")
                continue

            dict_st = text[result_st:].find(">{")
            dict_end = text[result_st:].find("}<")

            if dict_st == -1 or dict_end == -1:
                print(f"    Skipping {file.name} - no dictionary found")
                continue

            dict_str = text[result_st+dict_st+1:result_st+dict_end+1]
            dict_str = dict_str.replace("true", "True")
            dict_str = dict_str.replace("false", "False")

            log_dict = eval(dict_str)
            if log_dict["info"].get("score_norm", 0.0) > 1.0:
                log_dict["info"]["score_norm"] = 1.0
            logs.append(log_dict)

        except Exception as e:
            print(f"    Error processing {file.name}: {e}")
            continue

    print(f"    Successfully processed {len(logs)} files")
    return logs

def extract_word_counts(folder):
    word_counts = []
    files = list(folder.glob("*.out"))

    for i, file in enumerate(sorted(files)):
        try:
            text = file.read_text()
            p = 0
            wc = 0
            while p < len(text):
                msg_st_sentinal = "-------------------\n>[message] "
                msg_end_sentinal = "-------------------- "
                msg_st = text.find(msg_st_sentinal, p)
                if msg_st == -1:
                    break
                msg_end = text.find(msg_end_sentinal, msg_st)
                if msg_end == -1:
                    break
                msg = text[msg_st+len(msg_st_sentinal):msg_end]
                wc += len(msg.replace("\n", " ").split(" "))
                p = msg_end + 1
            word_counts.append(wc)

        except Exception as e:
            print(f"    Error processing word count for {file.name}: {e}")
            word_counts.append(0)
            continue

    return word_counts

print("Loading human logs...")
human_logs = []
with open(Path("/Users/ishangaur/projects/dialop/dialop/data/optimization.jsonl")) as f:
    for line in f:
        human_logs.append(json.loads(line.strip()))
print(f"Loaded {len(human_logs)} human logs")
human_scores = [log["result"]["norm"] for log in human_logs]
print(f"Extracted {len(human_scores)} human scores")
print("Calculating human word counts...")
human_word_counts = []
for log in human_logs:
    wc = 0
    for action in log["action_log"]:
        if "message" in action:
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

# Collect all data for plotting
all_scores = []
all_wcs = []
colors = ['blue', 'orange', 'green']
labels = []

for i, (label, folder) in enumerate(folders.items()):
    print(f"Processing {label}...")
    logs = extract_logs(folder)
    scores = [log["info"].get("score_norm", 0.0) for log in logs]
    wcs = extract_word_counts(folder)
    ax_main.scatter(wcs, scores, label=label, alpha=0.7, color=colors[i])
    all_scores.extend(scores)
    all_wcs.extend(wcs)
    labels.extend([label] * len(scores))
    print(f"  Found {len(logs)} logs for {label}")

ax_main.scatter(human_word_counts, human_scores, label="human", alpha=0.7, color=colors[2])
all_scores.extend(human_scores)
all_wcs.extend(human_word_counts)
labels.extend(['human'] * len(human_scores))

ax_main.set_xlabel("Word Count")
ax_main.set_ylabel("Score Normalized")
ax_main.legend()

# Top marginal (x-axis distribution)
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
for i, (label, folder) in enumerate(folders.items()):
    logs = extract_logs(folder)
    wcs = extract_word_counts(folder)
    ax_top.hist(wcs, bins=20, alpha=0.7, density=True, color=colors[i], label=label)
ax_top.hist(human_word_counts, bins=20, alpha=0.7, density=True, color=colors[2], label="human")
ax_top.set_ylabel("Density")
plt.setp(ax_top.get_xticklabels(), visible=False)

# Right marginal (y-axis distribution)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
for i, (label, folder) in enumerate(folders.items()):
    logs = extract_logs(folder)
    scores = [log["info"].get("score_norm", 0.0) for log in logs]
    ax_right.hist(scores, bins=20, alpha=0.7, density=True, orientation='horizontal', color=colors[i])
ax_right.hist(human_scores, bins=20, alpha=0.7, density=True, orientation='horizontal', color=colors[2])
ax_right.set_xlabel("Density")
plt.setp(ax_right.get_yticklabels(), visible=False)

plt.suptitle("Word Count vs Score with Marginal Distributions")
print("Saving plot...")
plt.savefig('score_vs_wordcount_with_marginals.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'score_vs_wordcount_with_marginals.png'")

# Print summary statistics
print(f"\nSummary:")
print(f"Human games: {len(human_logs)}")
for label, folder in folders.items():
    logs = extract_logs(folder)
    print(f"{label} games: {len(logs)}")
