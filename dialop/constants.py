from enum import Enum
from pathlib import Path
import json

DATA_FOLDER = Path("/home/ishangaur/dialop/data/")

class GameType(Enum):
    MATCHING = "optimization"
    PLANNING = "planning"
    MEDIATION = "mediation"

def get_data(human_user: bool, human_assistant: bool, game_type: GameType):
    if isinstance(game_type, str):
        try:
            game_type = GameType(game_type)
        except ValueError:
            raise ValueError(f"Invalid game type: {game_type}")

    if human_user and human_assistant:
        player_type_folder = DATA_FOLDER / "human-human"
        data_file = player_type_folder / f"{game_type.value}.jsonl"
        data = []
        with open(data_file, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    elif human_user and not human_assistant:
        player_type_folder = DATA_FOLDER / "human-lm"
        raise NotImplementedError("Human user with LM assistant setting not recorded yet")
    elif not human_user and human_assistant:
        raise NotImplementedError("LM user with human assistant setting not recorded yet")
    else:
        player_type_folder = DATA_FOLDER / "selfplay"
        raise NotImplementedError("Self-play setting not recorded yet")