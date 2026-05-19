import pandas as pd
from typing import List, Dict
from pathlib import Path


class ExcelLoader:
    SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".xlsm", ".xlsb", ".ods"}

    @staticmethod
    def _read_file(file_path: str) -> Dict[str, pd.DataFrame]:
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension not in ExcelLoader.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(ExcelLoader.SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported file type '{extension}'. Supported: {supported}")

        if extension == ".csv":
            return {path.stem: pd.read_csv(path)}

        if extension == ".tsv":
            return {path.stem: pd.read_csv(path, sep="\t")}

        return pd.read_excel(path, sheet_name=None)

    @staticmethod
    def load(file_path: str) -> List[Dict[str, str]]:
        """Load spreadsheet file and return list of dictionaries."""
        records = []
        for sheet_name, df in ExcelLoader._read_file(file_path).items():
            sheet_records = df.to_dict("records")
            for record in sheet_records:
                record["_sheet"] = sheet_name
            records.extend(sheet_records)
        return records

    @staticmethod
    def load_as_text(file_path: str) -> str:
        """Load spreadsheet file and convert to text format."""
        sections = []
        for sheet_name, df in ExcelLoader._read_file(file_path).items():
            sections.append(f"Sheet: {sheet_name}\n{df.to_string(index=False)}")
        return "\n\n".join(sections)
