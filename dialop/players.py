import json
import os
import pathlib
from rich.prompt import IntPrompt, Prompt
from rich.markup import escape
from openai import OpenAI

from envs import DialogueEnv
from utils import num_tokens

api_key = None
try:
    with open(pathlib.Path(__file__).parent / ".api_key") as f:
        x = json.load(f)
        # only need the api_key now
        # openai.organization = x["organization"]
        api_key = x["api_key"]
    print("Loaded .api_key")
except:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Warning: no OpenAI API key loaded.")

client = OpenAI(api_key=api_key)

class OutOfContextError(Exception):
    pass

class DryRunPlayer:

    def __init__(self, prompt, role, console, task="planning"):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.calls = 0
        self.task = task

    def observe(self, obs):
        self.prompt += obs

    def respond(self):
        self.calls += 1
        if self.role == "agent" and self.calls == 5:
            if self.task == "planning":
                return f" [propose] [Saul's, Cookies Cream, Mad Seoul]"
            elif self.task == "mediation":
                return f" [propose] User 0: [1], User 1: [15]"
        elif self.role == "user" and self.calls == 6:
            return f" [reject]"
        return f" [message] {self.calls}"

class LLMPlayer:

    def __init__(self, prompt, role, console, model_kwargs=None,
                 prefix="\nYou:", optional=None):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.optional = optional
        self.removed_optional = False
        self.model = "gpt-5-2025-08-07"
        self.model_kwargs = dict(
            model=self.model,
            reasoning={"effort": "high"},
        )
        if model_kwargs is not None:
            raise ValueError("model_kwargs is not supported")
        self.messages = [
            {
                "role": "developer",
                "content": prompt + ("" if optional is None else optional)
            },
            {
                "role": "user",
                "content": ""
            }
        ]

    def observe(self, obs):
        # skip the first observation so we don't have to add an explicit "developer" player
        # if obs[self.role] == self.messages[0]["content"]:
        #     return
        # if "[message]" not in obs[self.role]:
        #     self.messages[0]["content"] += "\n" + obs[self.role]
        # else:
        #     players = set(obs.keys()) - set("done") - set("turn_player")
        #     self.messages.append(
        #         {
        #             "role": str((players - obs["turn_player"])[0]), # observation came from the last player
        #             "content": obs[self.role]
        #         }
        #     )
        if "Partner" not in obs and self.messages[1]["content"] != '': # if it's the first message, it's just the observation without a role
            obs = "You: " + obs
        self.messages[1]["content"] += obs.strip() + "\n"

    def respond(self):
        self.console.rule(f"{self.role}'s turn")
        # remaining = 4096 - num_tokens(self.prompt)
        # if remaining < 0 and self.optional:
        #     self._remove_optional_context()
        #     remaining = 4096 - num_tokens(self.prompt)
        # Still out of context after removing
        # if remaining < 0:
        #     print("OUT OF CONTEXT! Remaining ", remaining)
        #     raise OutOfContextError()
        # kwargs = dict(
        #     prompt=self.prompt,
        #     max_tokens=min(remaining, 128),
        # )
        kwargs = dict(
            # max_tokens=128,
            input=self.messages
        )
        kwargs.update(**self.model_kwargs)
        response = client.responses.create(**kwargs)
        assert response.output[0].type == "reasoning"
        assert response.output[1].type == "message"
        assert len(response.output[1].content) == 1
        message = response.output[1].content[0].text.strip()
        self.console.print("Response: ", escape(message))
        if response.output[0].content:
            self.console.print("Reasoning: ", escape(response.output[0].content[0].text.strip()))
        # if response["choices"][0]["finish_reason"] == "length":
        #     if not self.optional:
        #         raise OutOfContextError()
        #     self._remove_optional_context()
        #     response = openai.Completion.create(**kwargs)
        #     self.console.print("Response: ",
        #                        escape(response["choices"][0]["text"].strip()))
        #     self.console.print("stop: ", response["choices"][0]["finish_reason"])
        # self.console.print(response["usage"])
        return message

    def _remove_optional_context(self):
        print("Cutting out optional context from prompt.")
        if self.removed_optional:
            print("!! already removed.")
            return
        self.prompt = (
            self.prompt[:self.prompt.index(self.optional)] +
            self.prompt[self.prompt.index(self.optional) + len(self.optional):])
        self.removed_optional = True

class HumanPlayer:

    def __init__(self, prompt, role, console, prefix="\nYou:"):
        self.prompt = prompt
        self.role = role
        self.console = console
        self.prefix = prefix

    def observe(self, obs):
        self.prompt += obs

    def respond(self):
        if not self.prompt.endswith(self.prefix):
            self.prompt += self.prefix
        self.console.rule(f"Your turn ({self.role})")
        self.console.print(escape(self.prompt))
        resp = ""
        if self.prefix.strip().endswith("You to"):
            id_ = Prompt.ask(
                escape(f"Choose a player to talk to"),
                choices=["0","1","all"])
            resp += f" {id_}:"
        mtypes = ["[message]", "[propose]", "[accept]", "[reject]"]
        choices = " ".join(
                [f"({i}): {type_}" for i, type_ in enumerate(mtypes)])
        type_ = IntPrompt.ask(
                escape(
                    f"Choose one of the following message types:"
                    f"\n{choices}"),
                choices=["0","1","2","3"])
        message_type = mtypes[type_]
        if message_type not in ("[accept]", "[reject]"):
            content = Prompt.ask(escape(f"{message_type}"))
        else:
            content = ""
        resp += f" {message_type} {content}"
        return resp
