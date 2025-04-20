import os
import pandas as pd
from typing import List, Dict
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# ====== Courtroom Agent Class ======
class CourtroomAgent:
    def __init__(self, name: str, system_prompt: str, model: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.name = name
        self.system_prompt = system_prompt.strip()
        self.history: List[Dict[str, str]] = []
        self.client = InferenceClient(model, token=os.getenv("HF_API_TOKEN"))

    def _format_prompt(self, user_msg: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_msg})
        prompt = ""
        for m in messages:
            prompt += f"<|{m['role']}|>\n{m['content']}\n"
        prompt += "<|assistant|>\n"
        return prompt

    def respond(self, user_msg: str, **gen_kwargs) -> str:
        prompt = self._format_prompt(user_msg)
        completion = self.client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            stream=False,
            **gen_kwargs
        )
        answer = completion.strip()
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": answer})
        return answer

# ====== System Prompts ======
DEFENSE_SYSTEM = """
You are Alex Carter, defense counsel.
Your job is to defend your client against the charges using legal reasoning, facts, and questioning.
"""

PROSECUTION_SYSTEM = """
You are Jordan Blake, prosecution attorney.
Your job is to argue the case against the defendant clearly and logically, using evidence and law.
"""

JUDGE_SYSTEM = """
You are Judge Taylor, presiding over this courtroom.
Your job is to give a fair verdict based on the arguments and evidence presented.
"""

PLAINTIFF_SYSTEM = """
You are Taylor Brooks, the plaintiff in this case.
You are here to present your grievances clearly and truthfully to the court.
"""

DEFENDANT_SYSTEM = """
You are Casey Morgan, the defendant in this case.
You are here to defend yourself by explaining your side of the story.
"""

WITNESS_SYSTEM = """
You are Jamie Lee, a key witness in this case.
Respond based only on what you've observed or experienced relevant to the case.
"""

# ====== Verdict Extraction ======
def extract_verdict(judge_text: str) -> int:
    judge_text = judge_text.lower()
    grant_keywords = [
        "in favor of plaintiff", "plaintiff wins", "granted", "rules for plaintiff",
        "rules in favor of plaintiff", "court supports the plaintiff"
    ]
    deny_keywords = [
        "in favor of defendant", "denied", "rules for defendant",
        "rules in favor of defendant", "plaintiff loses", "court rejects the plaintiff's claim"
    ]
    for keyword in grant_keywords:
        if keyword in judge_text:
            return 1
    for keyword in deny_keywords:
        if keyword in judge_text:
            return 0
    
    return 0

# ====== Main Simulation Function ======
def simulate_trial(case_text: str) -> int:
    defense = CourtroomAgent("Defense", DEFENSE_SYSTEM)
    prosecution = CourtroomAgent("Prosecution", PROSECUTION_SYSTEM)
    judge = CourtroomAgent("Judge", JUDGE_SYSTEM)
    witness = CourtroomAgent("Witness", WITNESS_SYSTEM)
    plaintiff = CourtroomAgent("Plaintiff", PLAINTIFF_SYSTEM)
    defendant = CourtroomAgent("Defendant", DEFENDANT_SYSTEM)

    prosecution.respond(f"Opening statement. Background: {case_text}")
    defense.respond("Opening statement in response.")
    prosecution.respond("Present arguments and question a witness.")
    defense.respond("Respond to prosecution and question the same witness.")
    witness.respond("Please provide your testimony based on the events of the case.")
    plaintiff.respond("Present your statement to the court.")
    defendant.respond("Present your defense to the court.")
    prosecution.respond("Give your closing statement.")
    defense.respond("Give your closing statement.")
    judge_text = judge.respond("Deliver the final verdict.")

    return extract_verdict(judge_text)

if __name__ == "__main__":
    input_df = pd.read_csv("sample_cases.csv")
    results = []
    for _, row in input_df.iterrows():
        case_id = row["id"]
        case_text = row["text"]
        verdict = simulate_trial(case_text)
        results.append({"id": case_id, "label": verdict})

    output_df = pd.DataFrame(results)
    output_df.to_csv("submission.csv", index=False)
    print("Submission file 'submission.csv' has been created.")
