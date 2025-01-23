import operator
from functools import reduce
from typing import Literal, Optional, Type

import numpy as np
import pandas as pd

from .column_filter import ColumnFilter
from .utils import HasDataframe


class DataFrameFilter(HasDataframe):
    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] | HasDataframe = None,
        *,
        track_description: bool = True,
    ) -> None:
        super().__init__(dataframe=dataframe)
        self.dependencies: dict[ColumnFilter, list[ColumnFilter]] = {}
        self.description: dict[int, tuple] = {}
        self.track_description = track_description

    def describe(self, filter: ColumnFilter) -> None:
        self.description[id(filter)] = (
            filter.column,
            *filter.describe(),
        )

    def column_name(self, filter: ColumnFilter) -> str:
        if (column := filter.column) == pd.Index:
            column_name = "index"
        elif isinstance(column, str):
            column_name = column.lower().replace("_", " ")
        else:
            column_name = str(column)
        return column_name

    def add(
        self, filter: ColumnFilter, *, dependencies: Optional[list[ColumnFilter]] = None
    ) -> None:
        filter.dataframe_filter = self
        if dependencies is None:
            dependencies = []
        self.dependencies[filter] = dependencies

    def get(
        self,
        column: str | Type[pd.Index],
        *,
        filter_type: Optional[
            Literal["quantile_range", "value_range", "selected_values"]
        ] = None,
    ) -> ColumnFilter:
        filters = [filter for filter in self.dependencies if filter.column == column]
        if len(filters) == 1:
            return filters[0]
        if filter_type is None:
            raise ValueError(
                f"multiple filters defined for column {column}, specify filter_type"
            )
        filters = [
            filter for filter in filters if getattr(filter, filter_type) is not None
        ]
        if len(filters) == 1:
            return filters[0]
        raise ValueError(
            f"multiple filters defined for column {column} and filter_type {filter_type}"
        )

    def describe_dependent(self, filter: ColumnFilter) -> None:
        if not self.track_description:
            return
        for _filter in self.dependencies:
            if filter not in self.dependencies[_filter]:
                continue
            if not _filter.is_active:
                _filter.reset()
            self.apply(track_description=True)

    def apply(
        self, max_dependency_depth: int = 100, track_description: Optional[bool] = None
    ) -> pd.DataFrame:
        def prefilter_dataframe(filter):
            nonlocal dataframe, selection
            selection_dependencies = [
                selection[id(dependency)]
                for dependency in self.dependencies[filter]
                if dependency.is_active
            ]
            if len(selection_dependencies) > 0:
                selection_dependencies = reduce(operator.and_, selection_dependencies)
                return dataframe.loc[selection_dependencies]  # type: ignore
            return dataframe

        if track_description is None:
            track_description = self.track_description
        dataframe = self.dataframe
        if dataframe is None:
            raise ValueError("dataframe not specified")
        num_active = len([filter for filter in self.dependencies if filter.is_active])
        applied = []
        selection = {}
        for filter in self.dependencies:
            if (
                len(
                    [
                        dependency
                        for dependency in self.dependencies[filter]
                        if dependency.is_active
                    ]
                )
                > 0
            ):
                continue
            with filter.on_dataframe(dataframe):
                if filter.is_active:
                    selection[id(filter)] = filter.selection
            applied.append(filter)
        depth = 0
        while True:
            if len(selection) == num_active:
                break
            if depth >= max_dependency_depth:
                raise StopIteration(
                    f"maximum dependency depth reached, active: {num_active}, selections: {len(selection)}"
                )
            for filter in self.dependencies:
                if not filter.is_active or filter in applied:
                    continue
                if any(
                    [
                        (dependency not in applied)
                        for dependency in self.dependencies[filter]
                        if dependency.is_active
                    ]
                ):
                    # still has unapplied active conditions
                    continue
                dataframe_prefiltered = prefilter_dataframe(filter)
                with filter.on_dataframe(dataframe_prefiltered):
                    _selection = filter.selection
                    selection_filtered = pd.Series(
                        np.repeat(False, len(dataframe)), index=dataframe.index
                    )
                    selection_filtered.loc[_selection[_selection].index] = True  # type: ignore
                    selection[id(filter)] = selection_filtered
                applied.append(filter)
            depth += 1
        if len(selection) > 0:
            dataframe_filtered = dataframe.loc[
                reduce(operator.and_, list(selection.values()))
            ]
        else:
            dataframe_filtered = dataframe
        if not track_description:
            return dataframe_filtered
        for filter in self.dependencies:
            dataframe_prefiltered = prefilter_dataframe(filter)
            with filter.on_dataframe(dataframe_prefiltered):
                self.describe(filter)
        return dataframe_filtered

    def update(
        self,
        selection: dict[str | Type[pd.Index], tuple],
        *,
        reset: bool = False,
        validate_selection: bool = False,
        reset_on_error: bool = False,
    ) -> pd.DataFrame:
        if reset:
            for filter in self.dependencies:
                filter.reset()
        for column, (filter_type, value) in selection.items():
            filter = self.get(column, filter_type=filter_type)
            filter.set(
                **{filter_type: value},
                validate_selection=validate_selection,
                reset_on_error=reset_on_error,
            )
        return self.apply(track_description=self.track_description)
