from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, Optional, Type

import numpy as np
import pandas as pd

from .utils import HasDataframe

if TYPE_CHECKING:
    from .dataframe_filter import DataFrameFilter


class ColumnFilter(HasDataframe):
    def __init__(
        self,
        column: str | Type[pd.Index],
        *,
        dataframe: Optional[pd.DataFrame | HasDataframe] = None,
        selected_values: Optional[Iterable[Any]] = None,
        value_range: Optional[tuple[int, int] | tuple[float, float]] = None,
        quantile_range: Optional[tuple[float, float]] = None,
        include_lower: bool = True,
        include_upper: bool = True,
    ):
        super().__init__(dataframe=dataframe)
        self.column = column
        self.dataframe_filter: Optional[DataFrameFilter] = None
        self.include_lower = include_lower
        self.include_upper = include_upper
        self._is_active: bool = False
        self._selected_values = None
        self._value_range = None
        self._quantile_range = None
        self.selected_values = selected_values
        self.value_range = value_range
        self.quantile_range = quantile_range

    @property
    def _invalid_filter_type_error(self) -> ValueError:
        return ValueError(
            "specify exactly one of 'selected_values', 'value_range' or 'quantile_range'"
        )

    @property
    def _no_dataframe_error(self) -> ValueError:
        return ValueError("dataframe not specified")

    @property
    def filter_type(
        self,
    ) -> Literal["selected_values", "value_range", "quantile_range"]:
        if self.selected_values is not None:
            return "selected_values"
        if self.value_range is not None:
            return "value_range"
        if self.quantile_range is not None:
            return "quantile_range"
        raise self._invalid_filter_type_error

    @property
    def value(self):
        if (filter_type := self.filter_type) == "selected_values":
            return self.selected_values
        elif filter_type == "value_range":
            return self.value_range
        else:
            return self.quantile_range

    @property
    def is_active(self) -> bool:
        return self._is_active

    @is_active.setter
    def is_active(self, is_active: bool) -> None:
        # if is_active == self.is_active:
        #     return
        self._is_active = is_active
        if self.dataframe_filter is not None:
            self.dataframe_filter.describe_dependent(self)

    @property
    def is_filtering(self) -> bool:
        if self.selected_values is not None:
            return len(self.selected_values) > 0
        if self.quantile_range is not None:
            return self.quantile_range != (0, 1)
        if self.value_range is not None:
            return self.value_range != (self.min, self.max)
        raise self._invalid_filter_type_error

    @property
    def selected_values(self) -> tuple[Any, ...] | None:
        return self._selected_values

    @selected_values.setter
    def selected_values(self, selected_values: Iterable[Any] | None) -> None:
        if selected_values is not None:
            selected_values = tuple(selected_values)
        if selected_values == self.selected_values:
            return
        self._selected_values = selected_values
        if selected_values is not None:
            self.value_range = None
            self.quantile_range = None
        self.is_active = selected_values is not None and len(selected_values) > 0

    @property
    def value_range(self) -> tuple[int, int] | tuple[float, float] | None:
        return self._value_range

    @value_range.setter
    def value_range(
        self, value_range: tuple[int, int] | tuple[float, float] | None
    ) -> None:
        if value_range == self.value_range:
            return
        self._value_range = value_range
        if value_range is not None:
            self.selected_values = None
            self.quantile_range = None
        self.is_active = value_range is not None and value_range != (self.min, self.max)

    @property
    def quantile_range(self) -> tuple[float, float] | None:
        return self._quantile_range

    @quantile_range.setter
    def quantile_range(self, quantile_range: tuple[float, float] | None) -> None:
        if quantile_range == self.quantile_range:
            return
        self._quantile_range = quantile_range
        if quantile_range is not None:
            self.selected_values = None
            self.value_range = None
        self.is_active = quantile_range is not None and quantile_range != (0, 1)

    @property
    def values(self) -> pd.Series | pd.Index:
        if self.dataframe is None:
            raise self._no_dataframe_error
        if self.column == pd.Index:
            return self.dataframe.index
        if not isinstance(self.column, str):
            raise ValueError("Column must be either pd.Index or a string")
        return self.dataframe.loc[:, self.column]

    @property
    def unique_values(self) -> np.ndarray:
        values = self.values
        # if np.isdtype(np.asarray(values).dtype, "bool"):
        #     return np.asarray([True, False])
        return np.unique(values)

    @property
    def min(self) -> int | float:
        values = self.values
        try:
            return values.min()  # type: ignore
        except TypeError:
            return np.asarray(values).min()

    @property
    def max(self) -> int | float:
        values = self.values
        try:
            return values.max()  # type: ignore
        except TypeError:
            return np.asarray(values).max()

    def quantiles(self, quantiles: Iterable[float]) -> tuple[int | float, ...]:
        values = self.values
        quantiles = np.asarray(quantiles).reshape(-1)
        try:
            assert not isinstance(values.dtype, pd.CategoricalDtype)
            return tuple(np.quantile(np.asarray(values), quantiles))
        except (AssertionError, TypeError, np._core._exceptions.UFuncTypeError):  # type: ignore
            if isinstance(values.dtype, pd.CategoricalDtype) and values.dtype.ordered:
                dtype = values.dtype
            else:
                dtype = pd.CategoricalDtype(categories=list(np.unique(values)), ordered=True)
            values = pd.Categorical(list(values), dtype=dtype)
            codes = values.codes
            return tuple(
                values.categories[np.quantile(np.asarray(codes), quantiles).astype(int)]
            )

    @contextmanager
    def on_dataframe(self, dataframe) -> Iterator[None]:
        _dataframe = self.dataframe
        self.dataframe = dataframe
        try:
            yield
        except Exception as e:
            raise e
        finally:
            self.dataframe = _dataframe

    @property
    def selection(self) -> pd.Series[bool]:
        selection_types = [self.selected_values, self.value_range, self.quantile_range]
        num_specified_selection_types = sum(
            [int(selection_type is not None) for selection_type in selection_types]
        )
        if num_specified_selection_types != 1:
            raise self._invalid_filter_type_error
        if self.dataframe is None:
            raise self._no_dataframe_error
        values = self.values
        if isinstance(values.dtype, pd.CategoricalDtype) and not values.dtype.ordered:
            dtype = pd.CategoricalDtype(categories=list(np.unique(values)), ordered=True)
            values = pd.Series(values, dtype=dtype)
        if self.selected_values is not None:
            if len(self.selected_values) == 0:
                return pd.Series(
                    np.repeat(True, len(values)), index=self.dataframe.index
                ).astype(bool)
            return pd.Series(
                np.isin(np.asarray(values), self.selected_values),
                index=self.dataframe.index,
            )
        value_range = None
        if self.value_range is not None:
            value_range = self.value_range
        if self.quantile_range is not None:
            value_range = self.quantiles(self.quantile_range)
        if value_range is None:
            raise self._invalid_filter_type_error
        # dtypes for value range, values and min and max correspond with each other
        lower: int | float = values.min()  # type: ignore
        upper: int | float = values.max()  # type: ignore
        if value_range[0] < lower:  # type: ignore
            value_range = lower, value_range[1]
        if value_range[1] > upper:  # type: ignore
            value_range = value_range[0], upper
        if value_range == (self.min, self.max):
            return pd.Series(np.repeat(True, len(values)), index=self.dataframe.index).astype(bool)
        above_lower = (
            (values >= value_range[0])
            if self.include_lower
            else (values > value_range[0])
        )
        below_upper = (
            (values <= value_range[1])
            if self.include_upper
            else (values < value_range[1])
        )
        return pd.Series(above_lower & below_upper, index=self.dataframe.index).astype(bool)

    @property
    def filtered_values(self) -> np.ndarray:
        if (selection := self.selection).all():
            return np.asarray(self.values)
        if isinstance(self.values, pd.Index):
            return np.asarray(self.values[selection])
        return np.asarray(self.values.loc[selection])

    def apply(self, dataframe=None) -> pd.DataFrame:
        if dataframe is None:
            dataframe = self.dataframe
        with self.on_dataframe(dataframe):
            if self.dataframe is None:
                raise self._no_dataframe_error
            if (selection := self.selection).all():
                return self.dataframe
            return self.dataframe.loc[selection]

    def describe(
        self, dataframe=None
    ) -> (
        tuple[Literal["quantile_range"], tuple[float, float], tuple[float, float]]
        | tuple[
            Literal["value_range"],
            tuple[float, float] | tuple[int, int],
            tuple[float, float] | tuple[int, int],
        ]
        | tuple[Literal["selected_values"], tuple[Any, ...], tuple[Any, ...]]
        | tuple[Literal["none"], None, None]
    ):
        if dataframe is None:
            dataframe = self.dataframe
        with self.on_dataframe(dataframe):
            if self.quantile_range is not None:
                return "quantile_range", (0, 1), self.quantile_range
            if self.value_range is not None:
                self.value_range = (
                    max(self.min, self.value_range[0]),
                    min(self.max, self.value_range[1]),
                )
                return "value_range", (self.min, self.max), self.value_range
            if self.selected_values is not None:
                unique_values = tuple(np.asarray(self.unique_values).tolist())
                selected_values = tuple(
                    [value for value in self.selected_values if value in unique_values]
                )
                self.selected_values = selected_values
                return (
                    "selected_values",
                    unique_values,
                    selected_values,
                )
            return "none", None, None

    def set(
        self,
        *,
        selected_values: Optional[Iterable[Any]] = None,
        value_range: Optional[tuple[float, float] | tuple[int, int]] = None,
        quantile_range: Optional[tuple[float, float]] = None,
        validate_selection: bool = False,
        reset_on_error: bool = False,
    ) -> None:
        selection_types = [selected_values, value_range, quantile_range]
        num_specified_selection_types = sum(
            [int(selection_type is not None) for selection_type in selection_types]
        )
        if num_specified_selection_types != 1:
            raise self._invalid_filter_type_error
        try:
            if selected_values is not None:
                if (
                    validate_selection
                    and not np.isin(
                        np.asarray(selected_values), self.unique_values
                    ).all()
                ):
                    raise AssertionError(f"invalid value selection: {selected_values}")
                self.selected_values = selected_values
                return
            if value_range is not None:
                if validate_selection and (
                    value_range[0] < self.min or value_range[1] > self.max
                ):
                    raise AssertionError(
                        f"invalid value range: {value_range} [min: {self.min}, max: {self.max}]"
                    )
                self.value_range = value_range
                return
            if quantile_range is not None:
                if validate_selection and (
                    quantile_range[0] < 0 or quantile_range[1] > 1
                ):
                    raise AssertionError(f"invalid quantile range: {quantile_range}")
                self.quantile_range = quantile_range
        except AssertionError as e:
            if reset_on_error:
                self.reset()
            else:
                raise e

    def reset(self) -> None:
        if self.selected_values is not None:
            self.selected_values = []
            return
        if self.value_range is not None:
            self.value_range = (self.min, self.max)
            return
        if self.quantile_range is not None:
            self.quantile_range = (0, 1)
