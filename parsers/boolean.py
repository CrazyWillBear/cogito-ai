from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser

# This code comes straight from the LangChain documentation
class BooleanOutputParser(BaseOutputParser[bool]):
    true_val: str = "YES"
    false_val: str = "NO"

    def parse(self, text: str) -> bool:
        cleaned_text = text.strip().upper()
        if cleaned_text not in (
            self.true_val.upper(),
            self.false_val.upper(),
        ):
            raise OutputParserException(
                f"BooleanOutputParser expected output value to either be "
                f"{self.true_val} or {self.false_val} (case-insensitive). "
                f"Received {cleaned_text}."
            )
        return cleaned_text == self.true_val.upper()

    @property
    def _type(self) -> str:
        return "boolean_output_parser"