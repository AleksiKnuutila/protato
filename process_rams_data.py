# %%

import json
import csv
from collections import Counter

# %%
def get_sentence(tokens, start_index):
    # Define sentence-ending punctuation
    sentence_endings = ['.', '?', '!']
    
    # Find the start of the sentence
    sentence_start = start_index
    while sentence_start > 0 and tokens[sentence_start - 1] not in sentence_endings:
        sentence_start -= 1
    
    # Find the end of the sentence
    sentence_end = start_index
    while sentence_end < len(tokens) - 1 and tokens[sentence_end] not in sentence_endings:
        sentence_end += 1
    
    # Include the ending punctuation in the sentence
    if sentence_end < len(tokens) and tokens[sentence_end] in sentence_endings:
        sentence_end += 1
    
    # Join the tokens to form the sentence
    return ' '.join(tokens[sentence_start:sentence_end])

# %%
def get_trigger_sentence(tokens, trigger_start, trigger_end):
    # Find the start and end of the sentence containing the trigger
    sentence_start = trigger_start
    while sentence_start > 0 and tokens[sentence_start - 1] not in ['.', '?', '!']:
        sentence_start -= 1
    
    sentence_end = trigger_end
    while sentence_end < len(tokens) and tokens[sentence_end] not in ['.', '?', '!']:
        sentence_end += 1
    
    # Include the ending punctuation in the sentence
    if sentence_end < len(tokens) and tokens[sentence_end] in ['.', '?', '!']:
        sentence_end += 1
    
    return ' '.join(tokens[sentence_start:sentence_end])

# %%
def load_and_analyze_jsonl(file_path, search_terms, output_csv):
    if isinstance(search_terms, str):
        search_terms = [search_terms]
    search_terms = [term.lower() for term in search_terms]
    
    matching_events = []
    all_argument_texts = []
    
    # List of roles to skip
    skip_roles = ['place', 'defendant', 'beneficiary', 'target', 'artifact', 'recipient', 'origin', 'destination', 'instrument', 'victim', 'otherparticipant']
    
    # First pass: collect all matching events
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            doc_id = data['doc_id']
            tokens = data['tokens']
            full_text = data['text']
            for event in data['event_mentions']:
                event_matched = False
                for argument in event['arguments']:
                    all_argument_texts.append(argument['text'])
                    if not event_matched and any(term in argument['text'].lower() for term in search_terms):
                        role = argument['role']
                        
                        if role.lower() in skip_roles:
                            continue
                        
                        event_type = event['event_type']
                        sentence = get_sentence(tokens, argument['start'])
                        entity_id = argument['entity_id']
                        argument_text = argument['text']  # Store the full argument text
                        
                        # Extract trigger sentence
                        trigger_start = event['trigger']['start']
                        trigger_end = event['trigger']['end']
                        trigger_sentence = get_trigger_sentence(tokens, trigger_start, trigger_end)
                        
                        matching_events.append({
                            'doc_id': doc_id,
                            'entity_id': entity_id,
                            'event_type': event_type,
                            'role': role,
                            'sentence': sentence,
                            'full_text': full_text,
                            'argument_text': argument_text,  # Add the argument text to the dictionary
                            'trigger_sentence': trigger_sentence
                        })
                        event_matched = True
                        break
    
    # Count role frequencies
    role_counter = Counter([event['role'] for event in matching_events])
    
    # Filter out events with roles that appear less than 5 times
    filtered_events = [event for event in matching_events if role_counter[event['role']] >= 5]
    
    # Write filtered events to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Doc ID', 'Entity ID', 'Role', 'Event Type', 'Sentence', 'Full Text', 'Argument Text', 'Trigger Sentence'])  # Add 'Argument Text' and 'Trigger Sentence' to the header
        
        for event in filtered_events:
            csvwriter.writerow([
                event['doc_id'],
                event['entity_id'],
                event['role'],
                event['event_type'],
                event['sentence'],
                event['full_text'],
                event['argument_text'],  # Include the argument text in the CSV output
                event['trigger_sentence']
            ])
    
    return filtered_events, all_argument_texts

# %%
# Function to analyze and print results
def analyze_and_print_results(matching_events, search_terms):
    # Analyze event types
    event_type_counter = Counter([event['event_type'] for event in matching_events])

    print(f"Event types featuring {search_terms} and their frequencies:")
    for event_type, count in event_type_counter.most_common():
        print(f"{event_type}: {count}")

    print(f"\nTotal number of events featuring {search_terms}: {len(matching_events)}")
    print(f"Number of unique event types featuring {search_terms}: {len(event_type_counter)}")

    # Analyze roles
    role_counter = Counter([event['role'] for event in matching_events])

    print(f"\nRoles of {search_terms} and their frequencies:")
    for role, count in role_counter.most_common():
        print(f"{role}: {count}")

    print(f"\nNumber of unique roles for {search_terms}: {len(role_counter)}")

    # Analyze event type and role combinations
    event_role_counter = Counter([(event['event_type'], event['role']) for event in matching_events])

    print(f"\nTop 20 event type and role combinations for {search_terms}:")
    for (event_type, role), count in event_role_counter.most_common(20):
        print(f"{event_type} - {role}: {count}")

# %%
# New function to analyze and print most common argument texts
def analyze_argument_texts(argument_texts, top_n=20):
    counter = Counter(argument_texts)
    print(f"\nTop {top_n} most common argument texts:")
    for text, count in counter.most_common(top_n):
        print(f"{text}: {count}")

# %%
# Set the file path
file_path = "/Users/aleksi/Apps/TextEE/data/processed_data/rams/all.jsonl"
output_csv = "rams_russia_events.csv"

# %%
# Example usage
search_terms = ["Russia"]  # You can change this to any string or list of strings
matching_events, all_argument_texts = load_and_analyze_jsonl(file_path, search_terms, output_csv)
analyze_and_print_results(matching_events, search_terms)
analyze_argument_texts(all_argument_texts)

print(f"\nResults have been saved to {output_csv}")

# %%
# You can easily analyze different terms by calling the functions again
# For example:
# search_terms = ["Tsarnaev"]
# output_csv = "tsarnaev_event_analysis.csv"
# matching_events, all_argument_texts = load_and_analyze_jsonl(file_path, search_terms, output_csv)
# analyze_and_print_results(matching_events, search_terms)
# analyze_argument_texts(all_argument_texts)

# %%
# Or multiple terms:
# search_terms = ["bomb", "explosion", "blast"]
# output_csv = "explosion_event_analysis.csv"
# matching_events, all_argument_texts = load_and_analyze_jsonl(file_path, search_terms, output_csv)
# analyze_and_print_results(matching_events, search_terms)
# analyze_argument_texts(all_argument_texts)

# %%
