import os
from openai import OpenAI
import pandas as pd
import re
import json

df = pd.read_csv('ProjectGutenberg-ShortStories-Dataset/stories.csv')

# Robust function to extract story content
def extract_story(text):
    # Step 1: Remove Gutenberg header/footer explicitly
    text = re.sub(r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*', '', text, flags=re.I|re.DOTALL)
    text = re.sub(r'End of (the|this) Project Gutenberg.*', '', text, flags=re.I|re.DOTALL)

    # Step 2: Remove production/transcriber notes and URLs
    text = re.sub(r'Produced by.*?(?=[A-Z])', '', text, flags=re.I|re.DOTALL)
    text = re.sub(r'\[.*?Transcriber.*?\]', '', text, flags=re.I|re.DOTALL)
    text = re.sub(r'http\S+', '', text)

    # Step 3: Normalize whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = text.replace('\\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 4: Identify the story's beginning
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    narrative_start = 0
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        # Heuristic: start from first sentence with >=8 words, ends with '.', '!', '?'
        if len(words) >= 8 and sentence[-1] in '.!?':
            narrative_start = i
            break
    
    story_text = ' '.join(sentences[narrative_start:]).strip()
    return story_text

print("Start")

# Apply extraction to 'content' column
df['story_only'] = df['content'].apply(extract_story)

print("Cleaned")

system_prompt = """Return the given story segmented into beginning, rising action, climax, and falling action in JSON format. Return the first sentence of the section followed by ... and then the last sentence of the section. Example: ```json
{
  "beginning": "“Oh, there IS one, of course, but you’ll never know it.”... Life’s too short for a ghost who can only be enjoyed in retrospect.",
  "rising_action": "But to the Boynes it was one of the ever-recurring wonders... she rose from her seat and stood among the shadows of the hearth.",
  "climax": "You knew about this, then--it’s all right?... I give you my word it never was righter!” he laughed back at her, holding her close.",
  "falling_action": "One of the strangest things she was afterward to recall... give it up, if that’s the best you can do.”"
}
```"""

annotated_stories = []

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

i = 1
CHAR_LIMIT = 32000  # Approx. 8000 tokens

for story in df['story_only'].iloc[22:]:
    if len(story) > CHAR_LIMIT:
        print(f"Skipping story {i}: too long ({len(story)} chars)")
        i += 1
        continue

    print(f"Processing story {i}...")

    try:
        response = client.responses.create(
            model="gpt-4o",
            instructions=system_prompt,
            input=story,
        )
        annotated_stories.append(response.output_text)

        with open('stories4.json', 'w') as f:
            json.dump(annotated_stories, f, indent=2)

    except Exception as e:
        print(f"Error on story {i}: {e}")

    i += 1

print("Finished")
