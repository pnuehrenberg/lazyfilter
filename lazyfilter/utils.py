from typing import Optional

import pandas as pd


class HasDataframe:
    def __init__(
        self, *, dataframe: Optional[pd.DataFrame] | "HasDataframe" = None
    ) -> None:
        self._dataframe = dataframe

    @property
    def dataframe(self) -> pd.DataFrame | None:  # type: ignore
        if self._dataframe is None or isinstance(self._dataframe, pd.DataFrame):
            return self._dataframe
        return self._dataframe.dataframe

    @dataframe.setter
    def dataframe(self, dataframe: Optional[pd.DataFrame] | "HasDataframe") -> None:
        self._dataframe = dataframe


class HasValidDataframe(HasDataframe):
    def __init__(self, *, dataframe: "pd.DataFrame | HasValidDataframe") -> None:
        self._dataframe = dataframe

    @property
    def dataframe(self) -> pd.DataFrame:
        if isinstance(self._dataframe, pd.DataFrame):
            return self._dataframe
        return self._dataframe.dataframe

    @dataframe.setter
    def dataframe(  # type: ignore
        self,
        dataframe: "pd.DataFrame | HasValidDataframe",
    ) -> None:
        self._dataframe = dataframe
