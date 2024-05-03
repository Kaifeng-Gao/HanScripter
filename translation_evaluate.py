from evaluate import load

def evaluate_translation(predictions, references):
    # Define the metrics to use
    metrics = {
        "sacrebleu": load("sacrebleu"),
        "meteor": load("meteor"),
        "chrf": load("chrf"),
        "bertscore": load("bertscore"),
    }
    
    # Dictionary to store results
    results = {}
    
    # Compute results for each metric
    for metric_name, metric in metrics.items():
        if metric_name == "bertscore":
            result = metric.compute(predictions=predictions, references=references, lang="en")
            average_precision = sum(result['precision']) / len(result['precision'])
            average_recall = sum(result['recall']) / len(result['recall'])
            average_f1 = sum(result['f1']) / len(result['f1'])
            average_scores = {
                'average_precision': average_precision,
                'average_recall': average_recall,
                'average_f1': average_f1
            }
            results[metric_name] = average_scores
        else:
            results[metric_name] = metric.compute(predictions=predictions, references=references)
    
    return results

def print_results(results):
    for metric_name, result in results.items():
        print(f"------------- {metric_name} results -------------")
        print(result)