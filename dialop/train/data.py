"""Utilities for turning logged dialogue games into training examples."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from dialop.constants import GameType, get_data


_OPT_INSTRUCTIONS = (
	Path(__file__).resolve().parent.parent / "envs" / "data" / "optimization.txt"
).read_text().strip()

_TASKS_SHORT = [
	"BLEU",
	"Electra",
	"GloVe",
	"GLUE",
	"LLaMA",
	"RoBERTa",
	"QuAC",
	"SWAG",
]

_WORKERS = [
	"Ava Li",
	"Daniel Nguyen",
	"Sofia Patel",
	"Andrei Petrov",
	"Morgan Reed",
	"Joseph Santos",
	"Ethan Smith",
	"Noah Wilson",
]


@dataclass(frozen=True)
class RenderedTurn:
	"""Lightweight representation of a single conversational turn."""

	player: int
	content: str
	turn_type: str


def _normalize_whitespace(text: str) -> str:
	"""Collapse repeated whitespace while preserving slash-separated tokens."""

	# Replace HTML breaks with spaces if they slipped into messages.
	text = text.replace("<br/>", " ").replace("<br>", " ")
	text = re.sub(r"\s+", " ", text)
	return text.strip()


def _render_proposal(raw_html: str) -> str:
	"""Turn stored HTML proposals into readable plain text."""

	text = unescape(raw_html)
	text = text.replace("<br/>", "\n").replace("<br>", "\n")
	text = text.replace("&emsp;", " ")
	lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
	lines = [line for line in lines if line]
	return "\n".join(lines)


def _render_action(action: Dict) -> Optional[RenderedTurn]:
	"""Map an action log entry to a uniform textual representation."""

	kind = action.get("type")
	if kind == "message":
		message = action["message"]["data"]
		return RenderedTurn(
			player=action["player"],
			content=f"[message] {_normalize_whitespace(message)}",
			turn_type=kind,
		)
	if kind == "proposal":
		proposal = _render_proposal(action["proposal"])
		return RenderedTurn(
			player=action["player"],
			content=f"[propose] {proposal}",
			turn_type=kind,
		)
	if kind == "proposal_response":
		token = "[accept]" if action["response"]["accept"] else "[reject]"
		return RenderedTurn(
			player=action["player"],
			content=token,
			turn_type=kind,
		)
	return None


def iter_matching_transcripts(records: Iterable[Dict]) -> Iterator[List[RenderedTurn]]:
	"""Yield cleaned message streams for optimization (matching) games."""

	for record in records:
		transcript: List[RenderedTurn] = []
		for action in record.get("action_log", []):
			rendered = _render_action(action)
			if rendered is not None:
				transcript.append(rendered)
		if transcript:
			yield transcript


def _format_table(table: List[List]) -> str:
	"""Format a table (with headers) into a simple ASCII view."""

	rows: List[str] = []
	for row in table:
		formatted_cells = []
		for cell in row:
			if cell == "":
				formatted_cells.append("-")
			else:
				formatted_cells.append(str(cell))
		rows.append(" | ".join(formatted_cells))
	return "\n".join(rows)


def _player_table(record: Dict, player: int) -> List[List[str]]:
	"""Reconstruct the per-player view of the similarity table."""

	table = record["table"]
	mask = record["mask1"] if player == 0 else record["mask2"]
	scale = record["scale1"] if player == 0 else record["scale2"]
	num_rows = len(table)
	num_cols = len(table[0]) if table else 0
	header = ["", *_TASKS_SHORT[:num_cols]]
	rows: List[List[str]] = [header]
	for i in range(num_rows):
		worker = _WORKERS[i] if i < len(_WORKERS) else f"Reviewer {i}"
		row: List[str] = [worker]
		for j in range(num_cols):
			if mask[i][j]:
				value = int(table[i][j] * scale)
				row.append(str(value))
			else:
				row.append("-")
		rows.append(row)
	return rows


def build_system_prompt(record: Dict, player: int) -> str:
	"""Combine instructions with the player's private table."""

	table_str = _format_table(_player_table(record, player))
	return (
		f"{_OPT_INSTRUCTIONS}\n\n"
		f"You are speaking as player {player}.\n"
		f"Here is your similarity table (unknown cells shown as '-'):\n"
		f"{table_str}\n"
		f"The best attainable total reward with full information is {record.get('best_assignment_reward', 'unknown')}.")


def iter_matching_turn_examples(
	records: Iterable[Dict],
	*,
	include_players: Iterable[int] = (0, 1),
) -> Iterator[Dict]:
	"""Yield supervised examples for each turn a target player speaks."""

	include_players = tuple(include_players)
	for convo_idx, record in enumerate(records):
		transcript: List[RenderedTurn] = []
		for action in record.get("action_log", []):
			rendered = _render_action(action)
			if rendered is not None:
				transcript.append(rendered)
		if not transcript:
			continue
		system_prompts = {
			player: build_system_prompt(record, player) for player in include_players
		}
		for player in include_players:
			history: List[Dict[str, str]] = []
			system_message = {"role": "system", "content": system_prompts[player]}
			for turn_idx, turn in enumerate(transcript):
				role = "assistant" if turn.player == player else "user"
				if turn.player == player:
					prompt_messages = [system_message, *history]
					yield {
						"id": f"matching-{convo_idx}-p{player}-t{turn_idx}",
						"conversation_id": convo_idx,
						"turn_index": turn_idx,
						"player": player,
						"messages": prompt_messages,
						"response": turn.content,
						"turn_type": turn.turn_type,
					}
				history.append({"role": role, "content": turn.content})


def load_matching_data() -> List[Dict]:
	"""Helper to load all human-human optimization games."""

	return get_data(human_user=True, human_assistant=True, game_type=GameType.MATCHING)


def export_matching_turns(records: Iterable[Dict], output_path: Path) -> int:
	"""Write per-turn training examples to JSON Lines."""

	count = 0
	with output_path.open("w", encoding="utf-8") as f:
		for example in iter_matching_turn_examples(records):
			f.write(json.dumps(example) + "\n")
			count += 1
	return count
