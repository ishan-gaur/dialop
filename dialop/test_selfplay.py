#!/usr/bin/env python3
"""
Simple self-play testing script for the optimization game.
Run with: python dialop/test_selfplay.py
"""

import json
import time
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich import print

from envs import OptimizationEnv
from players import LLMPlayer

def load_prompt(game, player):
    """Load prompt for a specific game and player role."""
    fname = f"{game}.txt"  # optimization uses same prompt for both players
    return (Path(__file__).parent / f"prompts/{fname}").read_text()

def save_game_log(game_data, log_dir):
    """Save game log to JSONL file in original format."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"selfplay_optimization_{timestamp}.jsonl"

    with open(log_dir / filename, 'a') as f:
        f.write(json.dumps(game_data) + '\n')

    return log_dir / filename

def run_selfplay_game(max_turns=30, game_num=1, log_dir=None):
    """Run a single self-play game between two LLM players."""
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
    for turn in range(max_turns):
        if obs.get("done", False):
            print(f"[bold blue]Game completed in {turn} turns![/bold blue]")
            break

        current_player = obs["turn_player"]
        player = players[current_player]

        print(f"\n[yellow]Turn {turn + 1}: {current_player}'s turn[/yellow]")

        try:
            # Get player's response
            response = player.respond()

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
            break

    # Print final results
    if obs.get("done", False):
        reward = obs.get("reward", 0)
        info = obs.get("info", {})
        print(f"\n[bold green]Final Results:[/bold green]")
        print(f"Reward: {reward}")
        print(f"Score: {info.get('score', 'N/A')}")
        print(f"Normalized Score: {info.get('score_norm', 'N/A'):.3f}")
        print(f"Total Messages: {info.get('num_msgs', 'N/A')}")
    else:
        print(f"\n[bold red]Game did not complete within {max_turns} turns[/bold red]")

    # Save game log in original format if directory provided
    if log_dir and obs.get("done", False):
        game_data = env.game.get_game_info()
        # Add result info
        info = obs.get("info", {})
        game_data["result"] = {
            "score": info.get("score", 0),
            "best": env.game.best_assignment_reward,
            "worst": 0,
            "norm": info.get("score_norm", 0)
        }
        log_file = save_game_log(game_data, log_dir)
        print(f"[dim]Game log saved to: {log_file}[/dim]")

    return obs

def main():
    """Run multiple self-play games."""
    console = Console()

    num_games = 3  # Start with a few games
    results = []

    # Create logs directory
    log_dir = Path("logs/selfplay_experiments")
    log_dir.mkdir(parents=True, exist_ok=True)

    for game_num in range(num_games):
        print(f"\n[bold cyan]{'='*50}[/bold cyan]")
        print(f"[bold cyan]Game {game_num + 1}/{num_games}[/bold cyan]")
        print(f"[bold cyan]{'='*50}[/bold cyan]")

        start_time = time.time()
        final_obs = run_selfplay_game(game_num=game_num + 1, log_dir=log_dir)
        elapsed = time.time() - start_time

        if final_obs.get("done", False):
            info = final_obs.get("info", {})
            results.append({
                "game_num": game_num + 1,
                "completed": True,
                "score": info.get("score", 0),
                "score_norm": info.get("score_norm", 0),
                "num_msgs": info.get("num_msgs", 0),
                "elapsed_time": elapsed
            })
        else:
            results.append({
                "game_num": game_num + 1,
                "completed": False,
                "elapsed_time": elapsed
            })

        print(f"\n[dim]Game {game_num + 1} finished in {elapsed:.1f}s[/dim]")

        # Brief pause between games
        if game_num < num_games - 1:
            time.sleep(2)

    # Print summary
    print(f"\n[bold magenta]{'='*50}[/bold magenta]")
    print(f"[bold magenta]SUMMARY[/bold magenta]")
    print(f"[bold magenta]{'='*50}[/bold magenta]")

    completed_games = [r for r in results if r["completed"]]

    if completed_games:
        avg_score = sum(r["score"] for r in completed_games) / len(completed_games)
        avg_score_norm = sum(r["score_norm"] for r in completed_games) / len(completed_games)
        avg_msgs = sum(r["num_msgs"] for r in completed_games) / len(completed_games)
        avg_time = sum(r["elapsed_time"] for r in completed_games) / len(completed_games)

        print(f"Completed: {len(completed_games)}/{num_games}")
        print(f"Average Score: {avg_score:.1f}")
        print(f"Average Normalized Score: {avg_score_norm:.3f}")
        print(f"Average Messages: {avg_msgs:.1f}")
        print(f"Average Time: {avg_time:.1f}s")
    else:
        print("No games completed successfully")

    # Print individual results
    print(f"\n[bold]Individual Results:[/bold]")
    for r in results:
        if r["completed"]:
            print(f"Game {r['game_num']}: Score={r['score']:.1f}, "
                  f"Norm={r['score_norm']:.3f}, Messages={r['num_msgs']}, "
                  f"Time={r['elapsed_time']:.1f}s")
        else:
            print(f"Game {r['game_num']}: INCOMPLETE, Time={r['elapsed_time']:.1f}s")

    print(f"\n[bold green]Game logs saved to: {log_dir}[/bold green]")

if __name__ == "__main__":
    main()
