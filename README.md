# 🧑‍⚖️ AI Courtroom Simulation

This is an intelligent courtroom simulation app built using **Streamlit**, **Hugging Face's Phi-3 model**, and a CSV dataset of legal cases. It simulates a realistic courtroom trial with AI agents playing roles such as Judge, Defense, Prosecution, Plaintiff, Defendant, and Witness.

---

## 📦 Features

- Simulates courtroom trials in multiple phases (Opening, Arguments, Witness Testimony, Closing, Verdict).
- Uses the Phi-3 model via Hugging Face Inference API to generate role-based responses.
- Allows you to load and choose from a list of legal cases.
- Provides a full **transcript** of the AI-generated trial.
- Automatically extracts a **verdict** (GRANTED or DENIED) from the Judge's response.

---

## 🛠️ Requirements

Install the required Python packages:

```bash
pip install streamlit pandas python-dotenv huggingface_hub
```

---

## 🔐 Setup

1. **Get a Hugging Face API Token**:
   - Create an account at [https://huggingface.co](https://huggingface.co)
   - Go to **Settings → Access Tokens** and generate a new token.

2. **Create a `.env` file** in your project folder and add:

   ```env
   HF_API_TOKEN=your_huggingface_token_here
   ```

3. **Prepare your dataset**:
   - Create a `sample_cases.csv` file with the following format:

     ```csv
     id,text
     1,"This is a legal dispute over property ownership..."
     2,"The plaintiff claims negligence by the defendant..."
     ```

---

## 🚀 How to Run

```bash
streamlit run app.py
```

- Use the **dropdown menu** to select a case.
- Click **▶️ Next Phase** to progress through the courtroom trial phases.
- Click **🔁 Restart Trial** to reset the simulation and agents.

---

## 🧠 AI Roles

Each role is powered by an instance of the `CourtroomAgent` class with custom system prompts:

- 👩‍⚖️ **Judge Taylor**: Delivers the final verdict.
- ⚖️ **Prosecution (Jordan Blake)**: Argues against the defendant.
- 🛡 **Defense (Alex Carter)**: Defends the accused.
- 🙋‍♂️ **Plaintiff (Taylor Brooks)**: States the complaint.
- 🙋‍♀️ **Defendant (Casey Morgan)**: Offers defense.
- 👁 **Witness (Jamie Lee)**: Provides testimony.

---

## 📜 Verdict Extraction

After the judge delivers their ruling, the app uses keyword-based logic to determine:

- **GRANTED** (1): if the judgment favors the **plaintiff**
- **DENIED** (0): if the judgment favors the **defendant**

---

## 📂 File Structure

```plaintext
├── app.py                # Main Streamlit app
├── sample_cases.csv      # Input case descriptions
├── .env                  # Contains Hugging Face API token
├── README.md             # Project documentation
```

---

## 🧪 Example Verdict Output

If the judge says:
> "The court rules in favor of the plaintiff. The motion is granted."

The verdict will be shown as:
```
Verdict: GRANTED
```

---


