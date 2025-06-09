# Pedestrian Attribute Recognition using Deep Learning

This project implements a pedestrian attribute recognition system using deep learning, based on the [PA-100K dataset](https://www.kaggle.com/datasets/yuulind/pa-100k). It includes models for clothing classification, accessories detection, age and gender prediction, and view estimation. Pedestrian detection is handled via YOLO from video input, allowing attribute recognition in real-world scenes.

The implementation is inspired by the paper:  
📄 [A Framework for Pedestrian Attribute Recognition Using Deep Learning](https://www.mdpi.com/2076-3417/12/2/622)  
And evaluated on the [PA-100K benchmark paper](https://arxiv.org/pdf/1709.09930).

⚡ Our models achieved a total accuracy of **88.29%**, surpassing the baseline reported in the HydraPlus paper.

---

## 🚀 Features

- **Modular Training Scripts**:
  - `clothing.py` — Upper, lower clothing & accessories (multi-task classification)
  - `accessories.py` — Standalone accessories model (multi-label classification)
  - `gender_age.py` — Joint age and gender prediction
  - `view.py` — Pedestrian view classification (front/back/side)
- **YOLO-based Evaluation**:
  - `eval.py` detects pedestrians in video input and applies trained models to extract attributes.
- **Testing Scripts**:
  - `test_clothing.py`, `test_demographics.py`, `test_view.py` — Evaluate respective models on the validation/test sets.
- **Utilities**:
  - Custom PyTorch Dataset (`ImageDatasetPT`), label group management, early stopping, and logging via `wandb`.

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/YasserRohaim/pedestrian-attribute-recognition.git
cd pedestrian-attribute-recognition
pip install -r requirements.txt
```

---

## 📁 Dataset

We use the [PA-100K dataset](https://www.kaggle.com/datasets/yuulind/pa-100k), which contains over 100,000 pedestrian images annotated with 26 attributes.

Download the dataset and place the images and annotation CSVs inside a `PA-100K/` directory as follows:

```
PA-100K/
├── data/
│   ├── <image files>
├── train.csv
├── val.csv
├── test.csv
```

---

## 🏋️‍♂️ Training

Train each model with the respective script:

```bash
# Clothing + Accessories classifier
python clothing.py

# Accessories-only classifier
python accessories.py

# Age & Gender classifier
python gender_age.py

# View (pose) classifier
python view.py
```

Each script uses `wandb` for tracking and includes early stopping, checkpointing, and metrics logging.

---

## 🧪 Testing

You can evaluate trained models using:

```bash
python test_clothing.py         # For clothing + accessories model
python test_demographics.py     # For age and gender model
python test_view.py             # For view estimation model
```

---

## 🎥 Real-time Evaluation

To detect pedestrians from a video and recognize their attributes using your trained models:

```bash
python eval.py --video path/to/video.mp4
```

YOLO is used to detect pedestrians, and each cropped region is passed through the trained models to infer demographic and clothing information.

---

## 🏆 Performance

| Task                   | Accuracy  |
|------------------------|-----------|
| View Estimation        | 81.61%    |
| Gender Classification  | 84.54%    |
| Age Classification     | 96.50%    |
| Upper Clothing         | 87.90%    |
| Lower Clothing         | 88.42%    |
| Accessories Detection  | 88.47%    |
| **Total Accuracy**     | **88.29%** ✅ |

✅ Outperforms the baseline of the [HydraPlus-Net](https://arxiv.org/pdf/1709.09930) on PA-100K.

---

## 📚 References

- [PA-100K Dataset](https://www.kaggle.com/datasets/yuulind/pa-100k)
- [PA-100K Original Paper (HydraPlus)](https://arxiv.org/pdf/1709.09930)
- [Main Implementation Paper (MDPI)](https://www.mdpi.com/2076-3417/12/2/622)

---

## 📬 Contact

Feel free to reach out or open issues for any questions or suggestions.