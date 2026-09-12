import os
import shutil
import tempfile

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.schemas.query import QueryResponse

from app.services.retrieval import (
    RetrievalService,
    get_retrieval_service,
)

from app.services.generation import (
    GenerationService,
    get_generation_service,
)


router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post(
    "/query",
    response_model=QueryResponse,
)
async def query(
    question: str = Form(...),
    image: UploadFile | str | None = File(None),
    retrieval: RetrievalService = Depends(
        get_retrieval_service
    ),
    generation: GenerationService = Depends(
        get_generation_service
    ),
):
    if not question.strip():
        raise HTTPException(
            status_code=422,
            detail="question must not be empty",
        )

    # Swagger UI sends an empty string for an omitted file upload rather
    # than truly omitting the field, so normalize that case to None here.
    if isinstance(image, str) or image is None:
        image = None

    # Retrieve the top 5 relevant chunks.
    results = retrieval.retrieve_chunks(
        question,
        top_k=5,
    )

    image_path = None
    tmp_path = None

    # Save uploaded image temporarily.
    if image is not None:
        suffix = os.path.splitext(
            image.filename or ""
        )[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tmp:
            shutil.copyfileobj(
                image.file,
                tmp,
            )

            tmp_path = tmp.name

        image_path = tmp_path

    try:
        answer, classification = (
            generation.generate_answer(
                question,
                results,
                image_path,
            )
        )

    finally:
        # Delete temporary uploaded image.
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    sources = [
        (
            f"{metadata['book']} "
            f"(pages "
            f"{metadata['page_start']}-"
            f"{metadata['page_end']})"
        )
        for metadata in results["metadatas"][0]
    ]

    image_classification = None

    if classification is not None:
        (
            dataset_key,
            predicted_class,
            confidence,
        ) = classification

        image_classification = {
            "dataset": dataset_key,
            "predicted_class": predicted_class,
            "confidence": confidence,
        }

    return QueryResponse(
        answer=answer,
        sources=sources,
        image_classification=image_classification,
    )