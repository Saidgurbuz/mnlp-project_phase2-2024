import json
from colorama import Fore, Style, init

init(autoreset=True)

def read_data(file_path):
    """ Read data from a JSONL file """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def analyze_results(results):
    """ Analyze results to count wins, losses, and ties """
    counts = {
        "A_wins": 0,
        "B_wins": 0,
        "ties": 0
    }

    for result in results:
        judgment = result['judgment']
        if '[[A]]' in judgment:
            counts["A_wins"] += 1
        elif '[[B]]' in judgment:
            counts["B_wins"] += 1
        elif '[[C]]' in judgment:
            counts["ties"] += 1

    total_judgments = len(results)
    win_rate_A = (counts["A_wins"] / total_judgments) * 100 if total_judgments > 0 else 0
    win_rate_B = (counts["B_wins"] / total_judgments) * 100 if total_judgments > 0 else 0
    tie_rate = (counts["ties"] / total_judgments) * 100 if total_judgments > 0 else 0

    return {
        "counts": counts,
        "rates": {
            "win_rate_A": win_rate_A,
            "win_rate_B": win_rate_B,
            "tie_rate": tie_rate
        }
    }

def main():
    results_file_path = 'judgment_results.json'
    judgments = read_data(results_file_path)
    
    # Analyze the results
    analysis = analyze_results(judgments)
    print(f"Results Analysis:\nCounts: {analysis['counts']}\nRates: {analysis['rates']}")

    print(f"{Fore.LIGHTGREEN_EX}Comparison and judgment completed successfully.{Style.RESET_ALL}")

if __name__ == '__main__':
    main()
