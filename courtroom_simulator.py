import os
import streamlit as st
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

# ====== App UI ======
st.set_page_config(page_title="Courtroom Simulator", layout="wide")
st.title("🧑‍⚖️ AI Courtroom Simulation")

# Load cases
@st.cache_data
def load_cases():
    return pd.read_csv("sample_cases.csv")

cases_df = load_cases()
case_titles = [f"Case {i+1}: {row['text'][:60]}..." for i, row in cases_df.iterrows()]

# Select case
case_index = st.selectbox("Select a Case to Simulate:", range(len(case_titles)), format_func=lambda i: case_titles[i])
selected_case = cases_df.iloc[case_index]
st.markdown(f"**Case Description:** {selected_case['text']}")

# Session state init
if "phase" not in st.session_state:
    st.session_state.phase = 0
    st.session_state.defense = CourtroomAgent("Defense", DEFENSE_SYSTEM)
    st.session_state.prosecution = CourtroomAgent("Prosecution", PROSECUTION_SYSTEM)
    st.session_state.judge = CourtroomAgent("Judge", JUDGE_SYSTEM)
    st.session_state.witness = CourtroomAgent("Witness", WITNESS_SYSTEM)
    st.session_state.plaintiff = CourtroomAgent("Plaintiff", PLAINTIFF_SYSTEM)
    st.session_state.defendant = CourtroomAgent("Defendant", DEFENDANT_SYSTEM)
    st.session_state.transcript = []

# Button controls
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔁 Restart Trial"):
        st.session_state.phase = 0
        st.session_state.defense = CourtroomAgent("Defense", DEFENSE_SYSTEM)
        st.session_state.prosecution = CourtroomAgent("Prosecution", PROSECUTION_SYSTEM)
        st.session_state.judge = CourtroomAgent("Judge", JUDGE_SYSTEM)
        st.session_state.witness = CourtroomAgent("Witness", WITNESS_SYSTEM)
        st.session_state.plaintiff = CourtroomAgent("Plaintiff", PLAINTIFF_SYSTEM)
        st.session_state.defendant = CourtroomAgent("Defendant", DEFENDANT_SYSTEM)
        st.session_state.transcript = []

with col2:
    if st.button("▶️ Next Phase"):
        st.session_state.phase += 1

# === Trial Phases ===
case_desc = selected_case['text']

if st.session_state.phase == 1:
    msg = st.session_state.prosecution.respond(f"Opening statement. Background: {case_desc}")
    st.session_state.transcript.append(("Prosecution", msg))
    msg = st.session_state.defense.respond("Opening statement in response.")
    st.session_state.transcript.append(("Defense", msg))

elif st.session_state.phase == 2:
    msg = st.session_state.prosecution.respond("Present arguments and question a witness.")
    st.session_state.transcript.append(("Prosecution", msg))
    msg = st.session_state.defense.respond("Respond to prosecution and question the same witness.")
    st.session_state.transcript.append(("Defense", msg))

elif st.session_state.phase == 3:
    msg = st.session_state.witness.respond("Please provide your testimony based on the events of the case.")
    st.session_state.transcript.append(("Witness", msg))

elif st.session_state.phase == 4:
    msg = st.session_state.plaintiff.respond("Present your statement to the court.")
    st.session_state.transcript.append(("Plaintiff", msg))
    msg = st.session_state.defendant.respond("Present your defense to the court.")
    st.session_state.transcript.append(("Defendant", msg))

elif st.session_state.phase == 5:
    msg = st.session_state.prosecution.respond("Give your closing statement.")
    st.session_state.transcript.append(("Prosecution", msg))
    msg = st.session_state.defense.respond("Give your closing statement.")
    st.session_state.transcript.append(("Defense", msg))

elif st.session_state.phase == 6:
    msg = st.session_state.judge.respond("Deliver the final verdict.")
    st.session_state.transcript.append(("Judge", msg))

# === Verdict Extraction (Grant/Denied) Based on Keywords ===
def extract_verdict(judge_text: str) -> int:
    judge_text = judge_text.lower()

    grant_keywords = [
        "in favor of plaintiff",
        "plaintiff wins",
        "granted",
        "rules for plaintiff",
        "rules in favor of plaintiff",
        "court supports the plaintiff"
    ]
    deny_keywords = [
        "in favor of defendant",
        "denied",
        "rules for defendant",
        "rules in favor of defendant",
        "plaintiff loses",
        "court rejects the plaintiff's claim"
    ]

    for keyword in grant_keywords:
        if keyword in judge_text:
            return 1
    for keyword in deny_keywords:
        if keyword in judge_text:
            return 0

    # Optional fallback
    if "plaintiff" in judge_text and "win" in judge_text:
        return 1
    if "defendant" in judge_text and "win" in judge_text:
        return 0

    return 0

# === Extract and Show Verdict (Only when Judge has spoken) ===
if st.session_state.phase >= 6 and len(st.session_state.transcript) > 0:
    last_speaker, judge_text = st.session_state.transcript[-1]
    if last_speaker == "Judge":
        verdict = extract_verdict(judge_text)
        st.session_state.verdict = verdict
        st.markdown(f"**Verdict:** {'GRANTED' if verdict == 1 else 'DENIED'}")

# === Show Transcript ===
st.divider()
st.subheader("📜 Trial Transcript")
for speaker, text in st.session_state.transcript:
    st.markdown(f"**{speaker}**: {text}")
