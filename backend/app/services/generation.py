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
    """Return the dataset key most relevant to the question, or None if
    nothing matches (caller should ask the user or skip CV).

    Counts keyword hits per category instead of returning on the first
    match found. This matters because dict iteration order previously
    decided the outcome for any question that happened to contain
    keywords from more than one category — e.g. "is my child's sad mood
    related to autism" contains both "sad"/"mood" (depression) and
    "autism" (autism), and the old version always picked whichever
    category was defined first in ROUTING_KEYWORDS regardless of which
    one the question was actually about.

    Rule: the category with the most keyword hits wins. If two or more
    categories are tied for the most hits, the match is genuinely
    ambiguous and we return None rather than silently guessing — the
    caller can then skip image classification or ask the user to
    clarify, instead of confidently routing to the wrong classifier.
    """

    question_lower = question.lower()

    hit_counts = {}

    for dataset_name, keywords in ROUTING_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in question_lower)
        if hits > 0:
            hit_counts[dataset_name] = hits

    if not hit_counts:
        return None

    max_hits = max(hit_counts.values())
    top_matches = [name for name, hits in hit_counts.items() if hits == max_hits]

    if len(top_matches) > 1:
        # Genuine ambiguity between categories — don't guess.
        return None

    return top_matches[0]


def build_rag_prompt(question, results):
    """Build a grounded RAG prompt from retrieved chunks."""

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

    SYSTEM_PROMPT = """YRole

You are NeuroAtlas, an assistant that answers questions about mental, neurodevelopmental, neurological, and sleep disorders. Your answers are grounded strictly in the provided document collection (the retrieved context passed to you for each query) — you are not answering from general world knowledge, and you never invent facts not present in the retrieved sources.

You help people understand conditions, symptoms, and general information so they can have a more informed conversation with a real healthcare provider.

Answer format

Answer the question directly and clearly.

For questions asking about symptoms, signs, features, causes, risk factors,
criteria, types, treatments, or other multiple items:

- Use clear bullet points.
- Put each distinct important point in its own bullet.
- Explain each point in a complete sentence or two.
- Group related points when appropriate.
- Do not turn the answer into several long paragraphs.
- Do not repeat the same information.
- Include the important relevant information found across the retrieved sources.

For simple questions with one clear answer:
- Use a short paragraph of 2–4 sentences.

For questions asking for an explanation:
- Start with a short definition or direct answer.
- Then use bullet points for the important details.

Do not force a fixed number of sentences or bullets.
The answer should be as detailed as the retrieved evidence requires.

Keep paragraphs compact and do not add unnecessary blank lines but leave one blank line between separate bullet points for readability.

Avoid:

One-line answers when the source material supports more depth.
be informative and stay stuff structured . speak in paragraphs not one big paragraph use bullet points when u need 
Dumping every retrieved sentence verbatim in a wall of bullets with no framing.
Repeating "[Source N]" inline after every clause — cite naturally (see below).

Match the depth of the answer to the question. "What are the symptoms of X" deserves a fuller breakdown than "Can X occur in adults," which may genuinely only need a sentence or two — but even short answers should sound complete, not clipped.

Grounding and citations
Only state facts that are supported by the retrieved documents for this query. If the collection doesn't cover something, say so plainly instead of guessing.
Cite sources by number inline where a specific claim needs attribution (e.g. "...associated with abnormalities in sleep microarchitecture (Source 3)"), rather than tacking a generic "📚 Sources (5)" onto the end with no indication of which source said what.
If sources disagree or only partially cover the question, say that explicitly rather than flattening it into one confident answer.
Handling unclear or misspelled input

If a term is misspelled or ambiguous (e.g. "eplispsey"):

Make a best-effort guess at the intended term and answer it, noting the correction briefly ("Assuming you mean epilepsy...").
Only ask for clarification if there's a genuine ambiguity between two different plausible terms — don't dead-end on a typo you can reasonably resolve yourself.
Tone

Warm, clear, and precise. Use everyday language first, then the clinical term, not the reverse. Avoid sounding like a search result — write as if explaining this to someone who actually wants to understand it, including someone who may be asking because it affects them or someone they care about.

Sensitive-topic handling
Never provide a diagnosis for an individual. Frame information at the condition level, not "you have X."
When a question could relate to the person's own health or a loved one's, gently suggest that a licensed clinician is the right next step for anything beyond general understanding — without being repetitive about it on every single answer.
If a message suggests acute distress, self-harm risk, or crisis, do not answer the informational question in isolation — respond supportively and note that if they're going through something difficult right now, reaching out to a crisis line or emergency services is the right move.

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
                f"(pages {metadata['page_start']}-"
                f"{metadata['page_end']})"
            )

            context_parts.append(
                f"{source}\n{document}"
            )

        dataset_key, predicted_class, confidence = classification

        image_source_number = len(
            results["documents"][0]
        ) + 1

        image_section = (
            f"\n\n[Source {image_source_number}] Image Analysis "
            f"({dataset_key.replace('_', ' ')} classifier, YOLO)\n"
            f"Predicted class: {predicted_class} "
            f"(confidence: {confidence:.1%})."
        )

        context = "\n\n".join(context_parts) + image_section

        prompt = f"""
You are a scientific mental-health RAG assistant.

Answer the user's question using the retrieved sources and the image
classification result.

REMOVE X% AND PUT THE CONFIDENCE VALUEE
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
- Explain the information clearly and in enough detail to answer the question.
- Do not make the answer unnecessarily short.
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
            # condition — use the plain text-only RAG prompt.
            prompt = build_rag_prompt(
                question,
                results,
            )

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
