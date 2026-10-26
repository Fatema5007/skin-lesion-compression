# Model compression for skin lesion classification: rare-class recall and robustness

Code for our ISBI 2027 submission (under review). We compare 75% pruning, post-training INT8 quantization, and
knowledge distillation, each about 4x smaller than a ResNet-50 (for pruning, 4x fewer non-zero weights), on HAM10000. We use a lesion-level split,
seven-fold cross-validation, two controls (fine-tune only, and a student trained without a teacher), and
corrupted test images (blur, brightness, noise).

## What is here
- `ISBI_code_cleaned.ipynb`: the whole pipeline (Google Colab, GPU).
  - Part 0: setup and data.
  - Part 1: first run on fold 0 with stronger corruptions (kept for transparency, not needed for the final tables).
  - Part 2: seven-fold training and evaluation of all six models.
  - Part 3: analysis (lesion bootstrap, paired differences), tables, and the robustness figure.
- `results/`: the printed output of the analysis cells (Cell D and Cell E), so the numbers in the paper can be checked.

## Data (not included)
Download HAM10000 yourself from the Harvard Dataverse (doi:10.7910/DVN/DBW86T) and read its terms of use.
Put these in one Google Drive folder, `MyDrive/RareClassRecall_SkinLesion/`:
`HAM10000_metadata.csv`, `ham10000_images_part_1/`, `ham10000_images_part_2/`.
Change `PROJECT_DIR` at the top of the notebook if you use another folder.

## How to run
1. Open the notebook in Colab and choose a GPU runtime.
2. Run Part 0, then Part 2, then Part 3. Part 1 is optional.
3. Part 3 writes `cv/results/*.csv`, `cv/figures/table*.tex`, and `cv/figures/fig1_robustness.pdf/png`.

Pretrained weights (ResNet-50, RegNetY-800MF) are downloaded by torchvision. The fold check in Part 2 compares fold 0 with the split of the first run, so use the same scikit-learn version as the one printed by the first cell of Part 0 (the split can change between versions). Part 2 took about 105 minutes in our run.

## Notes
- Each method was trained once per fold, so the intervals in the paper do not include training randomness.

## License
Code: MIT (see `LICENSE`). HAM10000 has its own terms, which apply to the data.
