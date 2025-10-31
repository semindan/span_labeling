"""
FastAPI server for constrained XML span labeling using pure HuggingFace generation.

Requirements:
    pip install "transformers==4.57.1" torch fastapi uvicorn sentencepiece
"""

import re
import warnings
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from span_labeling.config import get_hf_model, get_system_message
from span_labeling.hf_api_logits_processor import (
    XmlConstrainedLogitsProcessor,
    build_allowed_tags,
)
from span_labeling.prompt_utils import build_prompt, get_prompt_config

# Global variable to hold the loaded model and tokenizer
_model = None  # dict with keys: model, tokenizer


class AnnotationRequest(BaseModel):
    text: str
    method: str = "xml"  # Default to xml method
    dataset: str  # Required: dataset name (e.g., 'ner', 'error', 'multigec', etc.)
    debug: bool = False


class AnnotationResponse(BaseModel):
    input_text: str
    annotated_text: str
    full_output: str


def _get_eos_token_id(tokenizer) -> Optional[int]:
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return tokenizer.eos_token_id
    if getattr(tokenizer, "eos_token", None) is not None:
        try:
            return tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        except Exception:
            return None
    return None


def load_model():
    """Load the model once at startup."""
    global _model

    if _model is not None:
        return _model

    print("Loading model... This may take a while on first run.")

    # Suppress the resume_download deprecation warning
    warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

    model_name = get_hf_model()
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading model {model_name} with {torch_dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    # Ensure pad token exists for generation
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    _model = {"model": model, "tokenizer": tokenizer}

    print("Model and tokenizer loaded successfully!")
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown events."""
    # Startup: Load the model
    load_model()
    yield
    # Shutdown: cleanup if needed (nothing to do here)


# Create FastAPI app with lifespan handler
app = FastAPI(title="NER Annotation API", version="1.0.0", lifespan=lifespan)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "NER Annotation API is running",
        "model_loaded": _model is not None and _model.get("model") is not None,
    }


@app.post("/annotate", response_model=AnnotationResponse)
async def annotate(request: AnnotationRequest):
    """
    Annotate text with constrained XML tags using HuggingFace logits processor.

    Args:
        request: AnnotationRequest containing the text to annotate

    Returns:
        AnnotationResponse with the annotated text
    """
    if _model is None or _model.get("model") is None or _model.get("tokenizer") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.dataset:
        raise HTTPException(status_code=400, detail="Dataset is required")

    # Get prompt configuration to extract tag format and build allowed tags
    try:
        prompt_config = get_prompt_config(request.method, request.dataset)
        if not prompt_config:
            raise HTTPException(
                status_code=400,
                detail=f"No prompt configuration found for method '{request.method}' and dataset '{request.dataset}'",
            )
        allowed_openings, closing_tag = build_allowed_tags(prompt_config)
        # Build the user content using the existing utility
        entry = {"model_input": request.text}
        user_prompt = build_prompt(request.method, request.dataset, entry)
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid prompt configuration: {str(e)}"
        )

    try:
        model = _model["model"]
        tokenizer = _model["tokenizer"]

        # Prepare chat messages and apply the model's chat template
        messages = [
            {"role": "system", "content": get_system_message()},
            {"role": "user", "content": user_prompt},
        ]

        chat_inputs = None
        prompt_len = None

        # In debug mode, also show the non-tokenized chat-templated string
        if request.debug:
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                print("[DEBUG] Chat-formatted prompt:\n", formatted)
            except Exception as e:
                print("[DEBUG] Chat template stringify failed, falling back:", str(e))
                print("[DEBUG] Raw user prompt:\n", user_prompt)

        # Try to use chat template; if unavailable, fall back to plain prompt
        try:
            tokenized_chat = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )

            # tokenized_chat may be a Tensor or dict, normalize to dict with input_ids
            if isinstance(tokenized_chat, torch.Tensor):
                chat_inputs = {"input_ids": tokenized_chat.to(model.device)}
            elif isinstance(tokenized_chat, dict):
                chat_inputs = {k: v.to(model.device) for k, v in tokenized_chat.items()}
            else:
                # Unexpected type, fallback
                raise TypeError("Unexpected type returned by apply_chat_template")
            prompt_len = chat_inputs["input_ids"].shape[-1]
        except Exception:
            # Fallback: use raw user prompt (no chat template)
            if request.debug:
                print(
                    "[DEBUG] Falling back to raw prompt tokenization (no chat template)"
                )
            chat_inputs = tokenizer(user_prompt, return_tensors="pt")
            chat_inputs = {k: v.to(model.device) for k, v in chat_inputs.items()}
            prompt_len = chat_inputs["input_ids"].shape[-1]

        input_text_token_ids = tokenizer.encode(request.text, add_special_tokens=False)

        # Build constrained logits processor
        eos_id = _get_eos_token_id(tokenizer)
        processor = XmlConstrainedLogitsProcessor(
            tokenizer=tokenizer,
            input_text_token_ids=input_text_token_ids,
            allowed_opening_tags=allowed_openings,
            closing_tag=closing_tag,
            eos_token_id=eos_id,
            debug=request.debug,
        )
        logits_processor = [processor]

        # Size a safe upper bound for new tokens: proportional to input token count
        max_new_tokens = max(64, len(input_text_token_ids) * 3)
        if request.debug:
            print(
                f"[DEBUG] Generating with constrained decoding (max_new_tokens={max_new_tokens})"
            )

        gen_out = model.generate(
            **chat_inputs,
            do_sample=False,
            logits_processor=logits_processor,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
        )

        # Extract generated part (after prompt)
        new_tokens = gen_out[0][prompt_len:]
        annotated_text = tokenizer.decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # If the processor signaled a tail-bridging token with extra suffix, cut it off
        try:
            cut_suffix = (
                getattr(processor, "state", None).cut_suffix_chars
                if hasattr(processor, "state")
                and getattr(processor, "state", None) is not None
                else 0
            )
        except Exception:
            cut_suffix = 0
        if cut_suffix and cut_suffix > 0 and len(annotated_text) >= cut_suffix:
            if request.debug:
                print(
                    f"Cutting off {cut_suffix} trailing chars from annotated_text due to bridging token"
                )
            annotated_text = annotated_text[:-cut_suffix]

        # For transparency, return full output (prompt + generated)
        full_output = tokenizer.decode(
            gen_out[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        return AnnotationResponse(
            input_text=request.text,
            annotated_text=annotated_text,
            full_output=full_output,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Annotation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=5454, log_level="info")
