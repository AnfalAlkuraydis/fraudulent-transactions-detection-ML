# FictioNER 📖  
Fine-tuned [RoBERTa](https://huggingface.co/roberta-base) for Named Entity Recognition (NER) on the [LitBank](https://github.com/dbamman/litbank) dataset.

---

## 📌 Project Overview
This project trains a RoBERTa-based NER model to detect literary entities such as:

- **PER**: Persons  
- **GPE**: Geo-political entities  
- **LOC**: Locations  
- **FAC**: Facilities  
- **ORG**: Organizations  
- **VEH**: Vehicles  

The pipeline includes:
1. Parsing LitBank TSV annotations.  
2. Converting them to JSONL.  
3. Tokenizing with RoBERTa.  
4. Training with PyTorch + Hugging Face Transformers.  
5. Evaluation using `seqeval`.  
6. Interactive demo with **Gradio**.  

---

## 🚀 Results
- Training loss dropped steadily (≈0.31 → 0.15 → 0.10).  
- Validation F1 improved across epochs (0.74 → 0.77 → 0.78).  
- Test F1 ≈ **0.77 overall**, macro-F1 ≈ **0.64**.  
- **PER** entities are easiest (≈0.82–0.83 F1).  
- **ORG** remains hardest (few examples in dataset).  

---

## 🛠️ Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/AnfalAlkuraydis/FictioNER.git
cd FictioNER
pip install -r requirements.txt
```

---

## ▶️ Usage



### Run Gradio Demo
Launch the interactive demo:
```bash
python fictioner.py --demo
```

---

## 🎛️ Gradio Interface
Here’s an example of the interactive demo:

<img src="https://github.com/AnfalAlkuraydis/FictioNER/blob/main/gradio.PNG" width="800"/>

---

## 📝 Example Flagged Data
Feedback collected via Gradio flagging:

<img src="https://github.com/AnfalAlkuraydis/FictioNER/blob/main/flags.PNG" width="800"/>
