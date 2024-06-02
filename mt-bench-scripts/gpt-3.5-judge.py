import json
import gpt_wrapper
from gpt_wrapper.chat import Chat
from colorama import Fore, Style, init
import sys

init(autoreset=True)

def read_responses(file_path1, file_path2):
    """ Read responses from two JSONL files that are line-aligned """
    responses1 = []
    responses2 = []
    with open(file_path1, 'r', encoding='utf-8') as file1, open(file_path2, 'r', encoding='utf-8') as file2:
        for line1, line2 in zip(file1, file2):
            responses1.append(json.loads(line1)["response"])
            responses2.append(json.loads(line2)["response"])

    return responses1, responses2

def write_data(data, file_path):
    """ Write data to a JSONL file """
    with open(file_path, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')

def interact_and_finalize_response(chat_session, prompt_id, response1, response2, instruction):
    """ Send responses to a judging session and finalize the output """
    combined_text = f"[Prompt ID {prompt_id}]\nResponse 1: {response1}\nResponse 2: {response2}"
    print(f"{Fore.GREEN}Sending for judgment: {combined_text}{Style.RESET_ALL}")
    response = chat_session.ask(combined_text, instruction=instruction)
    return response.content, chat_session.chat_id

def compare_and_judge_responses(responses1, responses2):
    """ Compare responses from two models and use an AI judge to determine the best """
    gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com"
    gpt_wrapper.api_key = "8fd2fb5b-c73d-4f2d-b462-604105250d8b"

    results = []
    for i, (response1, response2) in enumerate(zip(responses1, responses2)):
        print(f"{Fore.MAGENTA}\nJudging responses for prompt index {i+1}:{Style.RESET_ALL}")
        chat_session = Chat.create(f"Judge_Chat_{i+1}")
        instruction = """
            Please act as an impartial judge and evaluate the quality of the responses provided by two
            AI assistants to the user question displayed below. You should choose the assistant that
            follows the user’s instructions and answers the user’s question better. Your evaluation
            should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
            and level of detail of their responses. Avoid any position biases and ensure that the
            order in which the responses were presented does not influence your decision. Do not allow
            the length of the responses to influence your evaluation. Do not favor certain names of
            the assistants. Be as objective as possible. Output your final verdict by STRICTLY following this format: "[[A]]" if assistant A is better, "[[B]]"
            if assistant B is better, and "[[C]]" for a tie. Only outputs "[[A]]", "[[B]]", or "[[C]]" are accepted"
        """
        judgment, chat_id = interact_and_finalize_response(
            chat_session, i+1, response1, response2, instruction)
        results.append({"prompt_index": i+1, "judgment": judgment, "chat_id": chat_id})
    return results

def main():
    file_path1 = "StefanKrsteski_Phi-3-mini-4k-instruct-DPO-EPFL_responses.jsonl"

    file_path2 = "microsoft_Phi-3-mini-4k-instruct_responses.jsonl"
    
    responses1, responses2 = read_responses(file_path1, file_path2)
    response_comparisons = compare_and_judge_responses(responses1, responses2)
    write_data(response_comparisons, 'judgment_results.json')
    print(f"{Fore.LIGHTGREEN_EX}Comparison and judgment completed successfully.{Style.RESET_ALL}")

if __name__ == '__main__':
    main()
