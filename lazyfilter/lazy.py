from collections.abc import Iterable
from typing import Optional, Type

import numpy as np
import pandas as pd

from .column_filter import ColumnFilter
from .dataframe_filter import DataFrameFilter
from .utils import HasDataframe


def lazy_filter(
    dataframe: pd.DataFrame | HasDataframe,
    *,
    dependencies: Optional[
        dict[str | Type[pd.Index], Iterable[str | Type[pd.Index]]]
    ] = None,
    track_description: bool = False,
    **dataframe_filter_kwargs,
):
    def get_columns(dataframe):
        if isinstance(dataframe, pd.DataFrame):
            return [pd.Index] + dataframe.columns.tolist()
        return get_columns(dataframe.dataframe)

    filters = {}
    for column in get_columns(dataframe):
        filter = ColumnFilter(column, dataframe=dataframe)
        filters[column] = filter
        dtype = filter.values.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            filter.selected_values = []
            continue
        if np.issubdtype(dtype, int) or np.issubdtype(dtype, float):  # type: ignore
            filter.value_range = (filter.min, filter.max)
            continue
        filter.selected_values = []
    if dependencies is None:
        dependencies = {}
    filter = DataFrameFilter(
        dataframe, track_description=track_description, **dataframe_filter_kwargs
    )
    for column in filters:
        _dependencies = []
        if column in dependencies:
            _dependencies = dependencies[column]
        _dependencies = [filters[dependency] for dependency in _dependencies]
        filter.add(filters[column], dependencies=_dependencies)
    return filter
