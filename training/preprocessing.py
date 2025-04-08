"""
Created on 30/03/2025

@author: Aryan

Filename: preprocessing.py
Relative Path: ./preprocessing.py

These functions preprocess different types of conversation datasets 
for training a tone-adaptive chatbot model.
"""


def preprocess_therapeutic_dataset(examples, tokenizer, max_length=512):
    """
    Preprocess therapeutic datasets from different sources into a unified format
    """
    inputs = []

    # Check for CounselChat structure
    if 'questionText' in examples and 'answerText' in examples:
        for q_text, a_text in zip(examples['questionText'], examples['answerText']):
            if not isinstance(q_text, str) or not isinstance(a_text, str):
                continue

            # Create a formatted conversation with therapeutic tone marker
            conversation = f"Human: {q_text}\nAssistant (therapeutic): {a_text}"
            inputs.append(conversation)

    # Check for EmpatheticDialogues structure
    elif 'conversations' in examples:
        for conversation in examples['conversations']:
            if not conversation:
                continue

            formatted_convo = ""
            for i, turn in enumerate(conversation):
                if i % 2 == 0:  # User turn
                    if isinstance(turn, dict) and 'content' in turn:
                        formatted_convo += f"Human: {turn['content']}\n"
                    elif isinstance(turn, dict) and 'role' in turn and turn['role'] == 'user' and 'content' in turn:
                        formatted_convo += f"Human: {turn['content']}\n"
                    elif isinstance(turn, str):
                        formatted_convo += f"Human: {turn}\n"
                else:  # Assistant turn
                    if isinstance(turn, dict) and 'content' in turn:
                        formatted_convo += f"Assistant (therapeutic): {turn['content']}\n"
                    elif isinstance(turn, dict) and 'role' in turn and turn['role'] == 'assistant' and 'content' in turn:
                        formatted_convo += f"Assistant (therapeutic): {turn['content']}\n"
                    elif isinstance(turn, str):
                        formatted_convo += f"Assistant (therapeutic): {turn}\n"

            if formatted_convo:
                inputs.append(formatted_convo.strip())

    # Check for situation/emotion structure
    elif 'situation' in examples and 'emotion' in examples:
        for situation, emotion in zip(examples['situation'], examples['emotion']):
            if not isinstance(situation, str) or not isinstance(emotion, str):
                continue

            # Create a formatted conversation with therapeutic tone marker
            conversation = f"Human: I'm feeling {emotion}. {situation}\nAssistant (therapeutic): I understand you're feeling {emotion}. Let's talk about your situation."
            inputs.append(conversation)

    # Check if inputs list is empty and return early
    if not inputs:
        # Return an empty dict with the expected structure
        return {"input_ids": [], "attention_mask": []}

    # Tokenize the inputs
    tokenized = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return tokenized

def preprocess_balanced_dataset(examples, tokenizer, max_length=512):
    """
    Preprocess balanced conversation datasets
    
    This handles structures like:
    - BlenderBot/DialogRPT datasets
    - ParlAI datasets with structured conversations
    """
    inputs = []

    # Handle ParlAI/blended_skill_talk structure
    if 'free_messages' in examples and 'guided_messages' in examples:
        for free_msgs, guided_msgs in zip(examples['free_messages'], examples['guided_messages']):
            conversation = ""

            # Interleave messages to create a conversation
            for i in range(min(len(free_msgs), len(guided_msgs))):
                if i % 2 == 0:
                    conversation += f"Human: {free_msgs[i]}\n"
                else:
                    conversation += f"Assistant (balanced): {guided_msgs[i]}\n"

            if conversation:
                inputs.append(conversation.strip())

    # Handle simple utterances structure
    elif 'utterances' in examples:
        for utterances_list in examples['utterances']:
            if isinstance(utterances_list, list):
                conversation = ""
                for i, utterance in enumerate(utterances_list):
                    if i % 2 == 0:
                        conversation += f"Human: {utterance}\n"
                    else:
                        conversation += f"Assistant (balanced): {utterance}\n"

                if conversation:
                    inputs.append(conversation.strip())
            elif isinstance(utterances_list, dict) and 'history' in utterances_list:
                # Handle persona-chat structure
                history = utterances_list['history']
                conversation = ""
                for i, message in enumerate(history):
                    if i % 2 == 0:
                        conversation += f"Human: {message}\n"
                    else:
                        conversation += f"Assistant (balanced): {message}\n"

                if conversation:
                    inputs.append(conversation.strip())

    # Tokenize the inputs
    tokenized = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return tokenized


def preprocess_casual_dataset(examples, tokenizer, max_length=512):
    """
    Preprocess casual conversation datasets
    
    This handles structures like:
    - DailyDialog format
    - Persona-Chat format
    """
    inputs = []

    # Handle DailyDialog format
    if 'utterances' in examples and isinstance(examples['utterances'][0], list):
        for utterances in examples['utterances']:
            conversation = ""
            for i, utterance in enumerate(utterances):
                if i % 2 == 0:
                    conversation += f"Human: {utterance}\n"
                else:
                    conversation += f"Assistant (casual): {utterance}\n"

            if conversation:
                inputs.append(conversation.strip())

    # Handle Persona-Chat format
    elif 'utterances' in examples and isinstance(examples['utterances'][0], dict) and 'history' in examples['utterances'][0]:
        for utterance_dict in examples['utterances']:
            history = utterance_dict.get('history', [])
            conversation = ""

            for i, message in enumerate(history):
                if i % 2 == 0:
                    conversation += f"Human: {message}\n"
                else:
                    conversation += f"Assistant (casual): {message}\n"

            if conversation:
                inputs.append(conversation.strip())

    # Tokenize the inputs
    tokenized = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return tokenized


def get_preprocessor_for_dataset_type(dataset_type):
    """Return the appropriate preprocessor function based on dataset type"""
    if dataset_type == "therapeutic":
        return preprocess_therapeutic_dataset
    elif dataset_type == "balanced":
        return preprocess_balanced_dataset
    elif dataset_type == "casual":
        return preprocess_casual_dataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def preprocess_and_tokenize_dataset(dataset, dataset_type, tokenizer, max_length=512):
    """Process and tokenize a dataset based on its type"""
    preprocessor = get_preprocessor_for_dataset_type(dataset_type)

    # Use batched processing for efficiency
    tokenized_dataset = dataset.map(
        lambda examples: preprocessor(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset
