import pandas as pd
from typing import List, Dict

class ExcelLoader:
    @staticmethod
    def load(file_path: str) -> List[Dict[str, str]]:
        """Load Excel file and return list of dictionaries."""
        df = pd.read_excel(file_path)
        return df.to_dict("records")

    @staticmethod
    def load_as_text(file_path: str) -> str:
        """Load Excel file and convert to text format."""
        df = pd.read_excel(file_path)
        return df.to_string()