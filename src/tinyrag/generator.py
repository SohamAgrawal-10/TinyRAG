from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "google/flan-t5-base"


class Generator:
    """Generate answers from a question and retrieved document context."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def generate(self, question: str, results: list[dict]) -> str:
        """Generate a grounded answer using retrieved passages."""

        context_parts = []

        for result in results:
            source = (
                f"{result['source']}:"
                f"{result['start_line']}-{result['end_line']}"
            )
            context_parts.append(
                f"[{source}]\n{result['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""Answer the question using only the provided context.

If the context does not contain enough information to answer the question,
say that the answer cannot be determined from the provided documents.

Context:
{context}

Question:
{question}

Answer:"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()

        sources = []
        for result in results:
            source = (
                f"{result['source']}:"
                f"{result['start_line']}-{result['end_line']}"
            )
            sources.append(source)

        return f"{answer}\n\nSources: {', '.join(sources)}"