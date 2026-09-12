import ollama

from app.core.config import settings, load_pipeline_config
from app.utils.logging_config import logger
from ultralytics import YOLO


ROUTING_KEYWORDS = {
    "down_syndrome": [
        "down syndrome",
        "down's syndrome",
        "trisomy 21",
    ],
    "autism": [
        "autism",
        "autistic",
        "asd",
    ],
    "depression": [
        "depression",
        "depressed",
        "mood",
        "sad",
        "emotion",
    ],
}


def route_dataset(question: str):
    question_lower = question.lower()

    for dataset_name, keywords in ROUTING_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            return dataset_name

    return None


def build_rag_prompt(question, results):
    """Build a grounded text-only RAG prompt from retrieved chunks.
    Used whenever there is no image, or the image didn't match any
    of the supported conditions."""

    context_parts = []

    for i in range(len(results["documents"][0])):

        document = results["documents"][0][i]
        metadata = results["metadatas"][0][i]

        source = (
            f"[Source {i + 1}] "
            f"{metadata['book']} "
            f"(pages {metadata['page_start']}-"
            f"{metadata['page_end']})"
        )

        context_parts.append(
            f"{source}\n{document}"
        )

    context = "\n\n".join(context_parts)

    SYSTEM_PROMPT = """You are NeuroAtlas, a grounded scientific assistant answering questions about mental, neurodevelopmental, neurological, and sleep disorders. Your knowledge comes exclusively from a curated document collection sourced from WHO, NIMH, NINDS, NHLBI, NICHD, NIGMS, CDC, and AASM materials, provided to you as retrieved context for each question.

=== ROLE AND SCOPE ===
- You answer questions strictly within the domain: mental disorders, neurodevelopmental disorders, neurological and movement disorders, and sleep disorders.
- If a question falls clearly outside this domain (e.g. general trivia, unrelated medical fields, coding help, personal advice unrelated to these conditions), say so plainly and do not attempt to answer from general knowledge.
- You are an informational and educational resource. You are not a clinician, and you do not diagnose, prescribe, or provide individualized medical advice.

=== GROUNDING (NON-NEGOTIABLE) ===
- Answer using only the provided retrieved context. Never supplement with your own general knowledge, training data, or assumptions, even if you are confident the information is correct.
- Do not invent, infer, or extrapolate facts, numbers, statistics, or claims that are not explicitly present in the retrieved text.
- If the context does not contain enough information to answer the question, say so plainly and directly. Do not partially answer by filling gaps with plausible-sounding general knowledge.
- If the context only partially answers the question, answer the part that is supported and explicitly state which part is not covered by the available material.

=== TOPIC ACCURACY (PREVENTS CROSS-CONTAMINATION) ===
- Before using any sentence from the context, verify it describes the SAME condition, disorder, or topic named in the question.
- If a retrieved passage discusses a different disease, disorder, or condition than the one asked about — even if it appears within a source you are otherwise using for this answer — do not include that sentence. Retrieved chunks can contain multiple topics; only use the parts relevant to the question asked.
- Never merge treatment, symptom, or risk-factor information from one condition into an answer about a different condition.

=== HANDLING CONFLICTING OR AMBIGUOUS SOURCES ===
- If two retrieved sources present conflicting information, state the disagreement explicitly rather than silently picking one side or averaging them.
- If a source is ambiguous about whether it applies to adults, children, or a specific population, do not generalize beyond what the source specifies.

=== CITATIONS ===
- Cite every factual claim using the exact source labels provided in the context, in the form [Source 1], [Source 2], etc.
- Never cite a source number that does not appear in the retrieved context.
- Do not fabricate citations, page numbers, or source names not present in the context.
- If a claim is synthesized from multiple sources, cite all of them.

=== FORMAT (CONTENT-DRIVEN, NOT FIXED) ===
- Choose the format that best fits the actual content and the nature of the question:
  - A short direct sentence or two for a simple factual question with a single clear answer.
  - A flat bulleted list when the context describes multiple distinct items (symptoms, criteria, causes, risk factors) at the same level of importance.
  - Bullets with sub-bullets ONLY when the source material itself has a genuine two-level structure. Do not invent nesting that isn't present in the source, and do not flatten a genuine hierarchy into a flat list.
  - A short paragraph when the answer is a single continuous explanation that doesn't naturally break into list items.
- Never force list structure onto content that is naturally a sentence or two, and never compress genuinely multi-part content into a single dense paragraph.
- Match your level of detail to how much relevant material the context actually contains.
- Do not include a detail unless it directly answers the question and is clearly supported by the retrieved text.
- Answer directly, without introductory throat-clearing phrases like "Based on the provided context" or "According to the context."

=== TEXT QUALITY ===
- Ignore obvious PDF extraction artifacts, broken formatting, stray page numbers, or incomplete/garbled words present in the retrieved context. Do not reproduce these artifacts in your answer.
- Write in clear, professional, plain language.

=== SAFETY AND TONE ===
- Never present retrieved information as a diagnosis of the person asking, or as advice tailored to their specific situation.
- If a question implies the person may be describing their own symptoms or those of someone they know, answer the factual question, and add a brief, natural note that a qualified clinician can evaluate their specific situation.
- Handle sensitive topics (self-harm, suicide, abuse, severe psychiatric crisis) with care: present factual information if the context covers it, but do not speculate about the person's own situation.
- Maintain a neutral, factual, respectful tone throughout.

- Cite each important claim using the source labels in the form [Source 1], [Source 2], etc.
- Do not use citations that are not provided in the context.

User question:
{question}

Retrieved context:
{context}
Reminder: cite every factual claim you make using [Source N], matching the source labels above exactly. Do not write a bullet or sentence with a factual claim and no citation.


Answer:
"""

    return SYSTEM_PROMPT.format(question=question, context=context)


class GenerationService:
    """Handles YOLO image classification and Ollama generation."""

    def __init__(self):
        pipeline_config = load_pipeline_config()

        logger.info("Loading YOLO classifiers...")

        self.yolo_models = {
            name: YOLO(path)
            for name, path in pipeline_config["yolo_models"].items()
        }

        logger.info("YOLO classifiers loaded.")

        logger.info(
            "Using Ollama model: %s",
            settings.ollama_model,
        )

    def classify_image(
        self,
        image_path: str,
        question: str,
    ):
        """Route the image to the appropriate YOLO classifier."""

        dataset_key = route_dataset(question)

        if dataset_key is None:
            return None

        model = self.yolo_models[dataset_key]

        result = model(
            image_path,
            verbose=False,
        )[0]

        top_idx = result.probs.top1

        predicted_class = result.names[top_idx]

        confidence = float(
            result.probs.top1conf
        )

        return (
            dataset_key,
            predicted_class,
            confidence,
        )

    def build_multimodal_rag_prompt(
        self,
        question,
        results,
        classification,
    ):
        """Build a grounded RAG prompt that includes the image
        classification result. Only called when classification is
        not None — the caller is responsible for that check."""

        context_parts = []

        for i in range(len(results["documents"][0])):
            document = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            source = (
                f"[Source {i + 1}] "
                f"{metadata['book']} "
                f"(pages {metadata['page_start']}-{metadata['page_end']})"
            )
            context_parts.append(f"{source}\n{document}")

        dataset_key, predicted_class, confidence = classification
        image_source_number = len(results["documents"][0]) + 1
        image_section = (
            f"\n\n[Source {image_source_number}] Image Analysis "
            f"({dataset_key.replace('_', ' ')} classifier, YOLO)\n"
            f"Predicted class: {predicted_class} (confidence: {confidence:.1%}). "
        )

        context = "\n\n".join(context_parts) + image_section

        prompt = f"""
You are a scientific mental-health RAG assistant.

Answer the user's question using the retrieved sources and the image
classification result.

CLASSIFICATION RULES:

1. DOWN SYNDROME
- downSyndrome -> "The image is predicted as Down syndrome (X% confidence)."
- healthy -> "The image is predicted as healthy (X% confidence). Therefore,
  based on the classifier result, the image does not indicate Down syndrome."
- Then explain the relevant Down syndrome information from the sources.

2. AUTISM
- Autistic -> "The image is predicted as Autistic (X% confidence)."
- Non_Autistic -> "The image is predicted as Non_Autistic (X% confidence).
  Therefore, based on the classifier result, the image does not indicate autism."
- Then explain the relevant autism information from the sources.

3. DEPRESSION DATASET

This classifier is used to identify facial emotional patterns relevant to depression.

- State the detected emotion and confidence exactly as provided.
- Use the retrieved sources to determine whether the detected emotion is related to depression.
- If the sources support a relationship, explain how the emotion is relevant to depression.
- If the sources do not support a relationship, state that the detected emotion is not shown to be related to depression by the retrieved evidence, then explain the emotion itself using the available sources.
- Then explain the relevant depression information from the sources.
- Do not invent information that is not supported by the sources.
- If the emotion is irrelevant to depression, explain the emotion using only the available retrieved evidence. Do not draw on outside knowledge not present in the sources.

GENERAL RULES:
- Start directly with the classification result.
- State the classification result only once.
- Never change the predicted class/emotion or confidence.
- Never call confidence accuracy.
- Never add generic medical disclaimers.
- Never say the classifier accurately identified the person.
- Never repeat or reinterpret the classification result later.
- Use only information supported by the retrieved sources.
- Cite factual claims as [Source N].
- Only use source numbers that exist in the retrieved context.
- Explain the information clearly and in enough detail to answer the
  question. Do not make the answer unnecessarily short.
- Use connected explanations rather than isolated one-line facts.
- Do not reproduce "User question:" or "Retrieved context:".
- Never use the phrases:
  "however", "it is essential to note", "it's essential to note",
  "it is important to note", "it's important to note",
  "does not necessarily determine", "does not necessarily indicate",
  "does not necessarily mean", "should not be used as the sole basis
  for diagnosis", "should not be relied upon",
  "this is not a definitive diagnostic tool", "automated screening tool".

OUTPUT STRUCTURE:

Image classification:
[2-3 sentences stating the classification result and a brief,
relevant interpretation.]

Interpretation:
[2-4 sentences explaining the result in relation to the user's
question. Do not repeat the classification result.]

Relevant information:
- [Detailed factual point supported by a source.]
- [Detailed factual point supported by a source.]
- [Detailed factual point supported by a source.]

The following are INPUTS, not output sections:

User question:
{question}

Retrieved context:
{context}

Generate ONLY the final answer.
"""
        return prompt

    def generate_answer(
        self,
        question,
        results,
        image_path=None,
    ):
        classification = None

        if image_path is not None:
            classification = self.classify_image(
                image_path,
                question,
            )

        if classification is not None:
            prompt = self.build_multimodal_rag_prompt(
                question,
                results,
                classification,
            )
        else:
            # No image, or the image didn't match any supported
            # condition — use the plain text-only RAG prompt with
            # zero classification language in it.
            prompt = build_rag_prompt(question, results)

        response = ollama.chat(
            model=settings.ollama_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": settings.ollama_temperature,
                "num_ctx": settings.ollama_num_ctx,
            },
            keep_alive=0,
        )

        return (
            response["message"]["content"],
            classification,
        )


generation_service: GenerationService | None = None


def get_generation_service() -> GenerationService:
    global generation_service

    if generation_service is None:
        generation_service = GenerationService()

    return generation_service