from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF, TER
from bert_score import score as bert_score_fn


def evaluate_results(predictions, references, split="train", device='cpu', tokenizer='13a'):
    """
    Evaluate prediction results using BLEU, ROUGE and BERTScore metrics.

    Args:
        predictions (list): List of predicted sequences.
        references (list): List of reference sequences.
        tokenizer (object, optional): Tokenizer if needed for evaluation.
        split (str): The data split being evaluated.
        device (str): Device for BERTScore computation.

    Returns:
        dict: A dictionary of evaluation scores.
    """
    log_dicts = {}

    bleu4 = BLEU(max_ngram_order=4, tokenize=tokenizer).corpus_score(predictions, [references]).score
    log_dicts[f"{split}/bleu4"] = bleu4

    if split == 'test':
        for i in range(1, 4):
            score = BLEU(max_ngram_order=i, tokenize=tokenizer).corpus_score(predictions, [references]).score
            log_dicts[f"{split}/bleu" + str(i)] = score

    if split != 'train':
        # ROUGE-L on val and test
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, pred)['rougeL'] for ref, pred in zip(references, predictions)]
        avg_f1 = sum(s.fmeasure for s in rouge_scores) / len(rouge_scores)
        log_dicts[f"{split}/rougeL_f1"] = avg_f1

        if split == 'test':
            log_dicts[f"{split}/rougeL_precision"] = sum(s.precision for s in rouge_scores) / len(rouge_scores)
            log_dicts[f"{split}/rougeL_recall"] = sum(s.recall for s in rouge_scores) / len(rouge_scores)

        # ChrF on val and test
        log_dicts[f"{split}/chrf"] = CHRF().corpus_score(predictions, [references]).score

        # BERTScore F1 on val and test
        _, _, F1 = bert_score_fn(predictions, references, lang='en', device=device, verbose=False)
        log_dicts[f"{split}/bertscore_f1"] = F1.mean().item()

    return log_dicts