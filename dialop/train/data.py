import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich import print
from dialop.envs import OptimizationEnv
from dialop.players import LLMPlayer
from dialop.constants import get_data, GameType

def load_prompt(game, player):
	"""Load prompt for a specific game and player role."""
	fname = f"{game}.txt"  # optimization uses same prompt for both players
	return (Path(__file__).parent.parent / f"prompts/{fname}").read_text()

def response_from_log(action_log_entry):
	# action types: message, proposal, proposal_response
	if action_log_entry["type"] == "message":
		return f" [message] {action_log_entry['message']['data']}"
	elif action_log_entry["type"] == "proposal":
		return f" [propose] {action_log_entry['proposal']}"
	elif action_log_entry["type"] == "proposal_response":
		if action_log_entry['response']['accept']:
			return " [accept]"
		else:
			return " [reject]"

def run_selfplay_game(action_log, max_turns=30):
	"""Run a single self-play game between two LLM players."""   
	dataset_observations = []
	console = Console()

	# Create environment
	env = OptimizationEnv()

	# Load prompts and create two LLM players
	prompt_text = load_prompt("optimization", "player")
	players = {
		"player-1": LLMPlayer(prompt_text, "player-1", console),
		"player-2": LLMPlayer(prompt_text, "player-2", console)
	}

	print("[bold green]Starting optimization self-play game...[/bold green]")

	# Reset environment and get initial observations
	obs = env.reset()

	# Give each player their initial observation
	for player_name, player in players.items():
		if player_name in obs:
			player.observe(obs[player_name])

	# Initialize action log for game data
	start_time = time.time()

	# Main game loop
	for turn in range(len(action_log)):
		if obs.get("done", False):
			print(f"[bold blue]Game completed in {turn} turns![/bold blue]")
			break

		current_player = obs["turn_player"]
		player = players[current_player]

		print(f"\n[yellow]Turn {turn + 1}: {current_player}'s turn[/yellow]")

		try:
			# Get player's response
			response = response_from_log(action_log[turn])
			print(response)
			# add current player's observation
			player_messages = deepcopy(players[current_player].messages)
			player_messages.append(
				{
					"role": "assistant",
					"content": response
				}
			)
			player_messages[0]['role'] = 'system'  # ensure first message is system
			dataset_observations.append(player_messages)

			# Step the environment
			obs, resample = env.step(response)


			if resample:
				print("[red]Error occurred, need to resample[/red]")
				continue

			# Update all players with new observations
			for player_name, player_obj in players.items():
				if player_name in obs and obs[player_name]:
					player_obj.observe(obs[player_name])

		except Exception as e:
			print(f"[red]Error during turn {turn + 1}: {e}[/red]")
			# Need to see how/when this happens, not sure how we want to handle this yet
			assert False, f"Error during turn {turn + 1}: {e}"

	# assertion for now, it is possible some of them didn't finish, but I need
	# to check how to verify that from the log
	if not obs.get("done", False):
		# seems like for some they just finished early
		if len(action_log) < max_turns:
			pass
		else:
			assert False, "Game did not complete successfully"
	return dataset_observations

def get_matching_human_human_sft_data(num_games=None):
	"""Load human-human matching data for SFT fine-tuning."""
	data = get_data(human_user=True, human_assistant=True, game_type=GameType.MATCHING)
	print(f"Loaded {len(data)} human-human matching games.")
	dataset = []
	if num_games is not None:
		data = data[:num_games]
	for i in range(len(data)):
		print(f"\n\n=== Running self-play for game {i+1}/{len(data)} ===")
		conversation_observations = run_selfplay_game(data[i]['action_log'])
		dataset.extend(conversation_observations)
	# tokenize (placeholder to set breakpoint below)
	print(len(dataset))
	  
if __name__ == "__main__":
	get_matching_human_human_sft_data(10)