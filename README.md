# P450HGT
Predicting horizontal gene transfer (HGT) candidates in cytochrome P450s with a kingdom classifier powered by a fine-tuned protein language model.

## Project Overview

**P450HGT** uses an ESM-1b protein language model fine-tuned on the **P450-90** dataset to encode each cytochrome P450 sequence into a **1280-dimensional** vector, and trains a multi-class **MLP** to predict the sequence’s **kingdom** label (bacteria, fungi, viridiplantae, metazoa, archaea, othereukaryote, virus). Sequences whose **predicted** labels disagree with their **annotated** labels are flagged as **putative horizontal gene transfer (HGT) candidates**. These candidates are then filtered with phylogenetic and genomic evidence to yield a high-confidence list of **bona fide HGT** genes.

## Methods at a Glance

### Data & Labels
- Dataset: **P450-90** with kingdom-level labels (**0–6**, seven classes).
- Split: **80/20** train/test.

### Representation Learning
- Fine-tune **ESM-1b** on P450-90 (see Supplementary).
- Encode each sequence into a **1280-d** numeric representation.

### Classification Model
- Architecture: **MLP 1280 → 256 → 256 → 7**  
- Loss: **Cross-Entropy**  
- Optimizer: **Adam (lr = 1e-4)**  
- Training: **100 epochs** to stable convergence

### Evaluation
- Metrics: **Precision / Recall / F1** (see Fig. S4 and Table S3).

### HGT Candidate & Criteria
- **Mismatch rule**: predicted kingdom ≠ annotated kingdom ⇒ mark as **HGT candidate**.  
- **Bona fide HGT** must satisfy **all three**:
  1. In a phylogenetic tree, the gene clusters **within the donor lineage/branch**.  
  2. The gene is **assembled on a genome** (not on short contigs).  
  3. The gene is **observed in at least three distinct species**.

## Key Features
- **Domain-specific embeddings**: P450-tuned ESM-1b better matches enzyme sequence distributions.  
- **Explainable screening**: kingdom-level discordance + phylogenomics reduces false positives.  
- **Reproducible training**: PyTorch implementation with fixed settings and standard metrics.

## Typical I/O
- **Input**: P450 protein sequences (FASTA), optional annotated kingdom labels.  
- **Output**: kingdom predictions per sequence, an **HGT candidate list** (based on label mismatch), and a **bona fide HGT** set after applying the three criteria.
