import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel
from collections import Counter

class Summaries(BaseModel):
    summaries: str

load_dotenv()

def truncate_event_type(event_type, frequencies, threshold=5):
    parts = str(event_type).split('.')
    
    # First round of truncation
    if frequencies[event_type] < threshold and len(parts) > 1:
        truncated = '.'.join(parts[:-1])
        return truncated
    
    return event_type

def double_truncate_event_types(df, threshold=5):
    # First round of truncation
    event_type_frequencies = df['Event Type'].value_counts()
    df['Truncated_Event_Type'] = df['Event Type'].apply(lambda x: truncate_event_type(x, event_type_frequencies, threshold))
    
    # Recalculate frequencies after first truncation
    truncated_frequencies = df['Truncated_Event_Type'].value_counts()
    
    # Second round of truncation
    df['Truncated_Event_Type'] = df['Truncated_Event_Type'].apply(lambda x: truncate_event_type(x, truncated_frequencies, threshold))
    
    return df

def extract_summaries(texts, doc_ids, dataset='rams', summary_length=5):
    summaries_file = f'extracted_summaries_{dataset}_{summary_length}words.csv'
    
    # Load the original data based on dataset
    if dataset == 'rams':
        df_original = pd.read_csv('rams_russia_events.csv')
    elif dataset == 'wiki':
        df_original = pd.read_csv('wikievents_actions_summaries.csv')
    else:
        raise ValueError("Dataset must be either 'rams' or 'wiki'")
    
    # Check if the summaries file already exists
    if os.path.exists(summaries_file):
        print(f"Loading existing summaries from {summaries_file}")
        df = pd.read_csv(summaries_file)
        existing_summaries = dict(zip(df['Doc ID'], df['Summary']))
        
        # Check if we need to process any new texts
        new_doc_ids = set(doc_ids) - set(existing_summaries.keys())
        if not new_doc_ids:
            print("All summaries already exist. No new processing needed.")
            return [existing_summaries[doc_id] for doc_id in doc_ids]
        
        df_to_process = df_original[df_original['Doc ID'].isin(new_doc_ids)]
    else:
        existing_summaries = {}
        df_to_process = df_original

    client = OpenAI()
    client.api_key = os.getenv("OPENAI_API_KEY")
    
    new_summaries = []
    for _, row in tqdm(df_to_process.iterrows(), desc="Extracting summaries", total=len(df_to_process)):
        if dataset == 'rams':
            summary = process_message_rams(client, row['Trigger Sentence'], row['Argument Text'], row['Full Text'], summary_length)
        else:  # wiki
            summary = process_message_wiki(client, row['Trigger Sentence'], row['Argument Text'], row['Full Text'], summary_length)
        new_summaries.append(summary)
    
    # Combine existing and new summaries
    all_summaries = existing_summaries.copy()
    all_summaries.update(dict(zip(df_to_process['Doc ID'], new_summaries)))
    
    # Save all summaries to CSV
    df_to_save = pd.DataFrame({
        'Doc ID': all_summaries.keys(),
        'Summary': all_summaries.values()
    })
    df_to_save.to_csv(summaries_file, index=False)
    print(f"Saved all summaries to {summaries_file}")
    
    # Return summaries in the original order
    return [all_summaries[doc_id] for doc_id in doc_ids]

def process_message_rams(client, trigger_sentence, argument_text, full_text, summary_length=5):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert observer of the actions of Russia that can explain specifically Russia's actions briefly."},
            {"role": "user", "content": f"Provide a summary of Russian government's actions as they are explicitly mentioned in the following text. More precise instructions are:\
            - The most important thing is this: Focus on what groups affiliated with Russia are doing. Don't summarise the actions of any other parties in the sentence, under any circumstances.\n\
            - The summary should be {summary_length} words, describing specifically what Russia is doing.\n\
            - Focus specifically and exclusively on what Russia is doing, not the overall activity in the text. Don't describe how others are reacting to Russia's actions.\n\
            - In your summary, don't mention individual locations or people by name, or name any groups such as particular nationalities. Describe the action and its object in general terms. Also don't mention Russia as the subject. I am interested in the type of activity, not who it is directed at or who is doing it.\n\
            - Also if the text states that Russia is not doing something (negative statement), just mention the action in a positive form. Again, I am interested in the type of activity discussed in the text, or what the text is about, not whether it is stated to be happening or not.\n\
            - If the text is using figurative language, try to describe Russia's action as concretely as possible.\n\
            - If the text describes many activities by Russia, focus your summary on the most concrete and tangible activity.\n\
            - If the text discusses failing to do something or not doing something, just mention the action in a positive form. Again, I am interested in the type of activity discussed in the text, or what the text is about, not whether it is stated to be happening or not.\n\
            - Use second person past tense in summary, first letter capitalised, no dot at the end.\n\
            - Focus only on actions directly attributed to Russia in the text, without any speculation or external information.\n\n\
            I need you to describe what the actor identified as \"{argument_text}\" is doing in the following sentence: {trigger_sentence}\n\n\
            The following segments provide some context for the sentence above. Take this into account, but describe specifically what Russia is doing in the sentence given above:\n\n{full_text}"}
        ],
        response_format=Summaries,
    )
    return completion.choices[0].message.parsed.summaries

def process_message_wiki(client, trigger_sentence, argument_text, full_text, summary_length=5):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in analyzing and summarizing event descriptions."},
            {"role": "user", "content": f"Provide a {summary_length}-word summary of the action described in the following text, focusing on what the actor is doing. The actor is identified as \"{argument_text}\" in this sentence: {trigger_sentence}\n\n\
            Context for the sentence:\n\n{full_text}"}
        ],
        response_format=Summaries,
    )
    return completion.choices[0].message.parsed.summaries