"""
Prompt templates for the generation module.

This module defines prompts used to control LLM behavior in the RAG pipeline.
The prompts enforce grounded generation and reduce hallucinations by restricting
the model to retrieved documents.
"""


def answer_prompt(query: str, context: str, history: str) -> str:
    return f"""
You are a document-grounded assistant.

Follow these rules STRICTLY:
1. Answer using ONLY the information from the "Documents" section.
2. Base your answer on specific statements from the documents.
3. Paraphrase the information, but DO NOT introduce new ideas.
4. If a statement is not clearly supported by the documents, DO NOT include it.
5. Prefer slightly incomplete but correct answers over general or guessed ones.
6. If the documents are not relevant, respond EXACTLY with:
   "I could not find the answer in the documents."
7. Keep the answer concise (max 2-3 sentences).

Conversation History:
{history}

Question:
{query}

Documents:
{context}

Answer:
""".strip()


def summary_prompt(topic: str, context: str) -> str:
    return f"""
You are a document-grounded assistant creating concise study notes.

You MUST follow these rules strictly:
1. Use ONLY the information from the "Documents" section.
2. Do NOT add any external knowledge.
3. Do NOT guess or hallucinate.
4. If the documents do not contain enough information, respond EXACTLY with:
   "I could not find enough information in the documents."
5. Write clearly and concisely.

Return the summary in EXACTLY three sections:

Definition:
- A clear explanation of the concept

Key Points:
- Bullet points with the most important information

Practical Importance:
- Why this concept matters in the given document context

Topic:
{topic}

Documents:
{context}

Summary:
""".strip()