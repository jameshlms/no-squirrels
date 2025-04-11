try:
    import firebase_admin
    from firebase_admin import firestore
    from google.cloud.firestore_v1 import FieldFilter, CollectionReference, Query

    FIREBASE_ADMIN_AVAILABLE = True

except ImportError as e:
    firebase_admin = None
    firestore = None
    FIREBASE_ADMIN_AVAILABLE = False

from typing import Any, Generator
from pandas import DataFrame
from warnings import warn


def _add_index(
    index: str,
    indices: list[str],
    existing: set[str],
) -> None:
    if not isinstance(index, str):
        raise TypeError(f"index must be a string. Got instead {type(index)}.")
    if not isinstance(indices, list):
        raise TypeError(f"indices must be a list. Got instead {type(indices)}.")
    if not isinstance(existing, set):
        raise TypeError(f"existing must be a set. Got instead {type(existing)}.")

    if index in existing:
        raise ValueError(
            f"Duplicate index found: {index}. Please provide a unique row_key."
        )
    indices.append(index)
    existing.add(index)


def _check_arg_type(
    arg: Any,
    arg_name: str,
    expected: type | tuple[type, ...],
    allow_none: bool = False,
) -> None:
    if allow_none and arg is None:
        return
    if not isinstance(arg, expected):
        if isinstance(expected, tuple):
            expected_str = " or ".join([e.__name__ for e in expected])
        else:
            expected_str = expected.__name__
        raise TypeError(
            f"{arg_name} must be of type {expected_str}. Got instead {type(arg).__name__}."
        )


def _check_firebase_admin_available() -> None:
    if not FIREBASE_ADMIN_AVAILABLE:
        raise ImportError("the firebase_admin package is not installed.")


def df_from_firestore(
    app_or_client,
    collection: str,
    *,
    filter: FieldFilter = None,
    limit: int = None,
    key_as_index: str | None = None,
    doc_id_as_index: bool = False,
    doc_id_column: str | None = None,
    flatten_arrays: bool = False,
    flatten_hashmaps: bool = True,
) -> DataFrame:
    """
    Fetch data from Firestore and return it as a DataFrame.

    Args:
        app: Firestore app instance.
        collection: Name of the collection to filter.
        filter: filter to execute.
        limit: Maximum number of documents to return.

    Returns:
        List of documents matching the filter.
    """
    # Base checks
    _check_firebase_admin_available()
    _check_arg_type(app_or_client, "app", (firebase_admin.App, firestore.Client))
    _check_arg_type(collection, "collection", str)
    _check_arg_type(filter, "filter", FieldFilter, allow_none=True)
    _check_arg_type(limit, "limit", int, allow_none=True)
    _check_arg_type(key_as_index, "key_as_index", str, allow_none=True)
    _check_arg_type(doc_id_as_index, "doc_id_as_index", bool)
    _check_arg_type(doc_id_column, "doc_id_column", str, allow_none=True)
    _check_arg_type(flatten_arrays, "flatten_arrays", bool)
    _check_arg_type(flatten_hashmaps, "flatten_hashmaps", bool)

    db = (
        firestore.client(app_or_client)
        if isinstance(app_or_client, firebase_admin.App)
        else app_or_client
    )
    docs = db.collection(collection)

    if limit is not None:
        docs = docs.limit(limit)

    if filter is not None:
        docs = docs.where(filter=filter)

    indices = []
    existing_indices = set()
    records = []

    for doc in docs.stream():
        if not doc.exists:
            continue

        record: dict[str | Any] = doc.to_dict()

        if key_as_index is not None:
            if doc_id_as_index:
                raise ValueError(
                    "key_as_index cannot be specified when doc_id_as_index is True."
                )
            try:
                _add_index(record[key_as_index], indices, existing_indices)
            except KeyError:
                raise KeyError(
                    f"key_as_index '{key_as_index}' not found in document {doc.id}."
                )
            if doc_id_column is not None:
                record[doc_id_column] = doc.id
        else:
            if doc_id_column and doc_id_as_index:
                warn(
                    f"The Document's IDs will appear twice as index and in column '{doc_id_column}'.",
                    UserWarning,
                )
            if doc_id_column is not None:
                record[doc_id_column] = doc.id
            if doc_id_as_index:
                _add_index(doc.id, indices, existing_indices)

        records.append(record)

    return DataFrame.from_records(
        records,
        index=indices if indices else None,
    )


class FirestoreParser:

    def __init__(self, app_or_client):
        _check_firebase_admin_available()
        _check_arg_type(app_or_client, "app", (firebase_admin.App, firestore.Client))
        self.db = (
            firestore.client(app_or_client)
            if isinstance(app_or_client, firebase_admin.App)
            else app_or_client
        )

    def set_collection(self, collection: str) -> None:
        _check_arg_type(collection, "collection", str)
        self.collection = self.db.collection(collection)

    def get_data_frame(
        self,
        *,
        collection: str | None = None,
        filter: FieldFilter = None,
        limit: int = None,
        key_as_index: str | None = None,
        doc_id_as_index: bool = False,
        doc_id_column: str | None = None,
        flatten_arrays: bool = False,
        flatten_hashmaps: bool = True,
    ) -> DataFrame:
        _check_arg_type(collection, "collection", str, allow_none=True)
        if collection is not None:
            self.set_collection(collection)

        return df_from_firestore(
            app_or_client=self.db,
            collection=self.collection.id,
            filter=filter,
            limit=limit,
            key_as_index=key_as_index,
            doc_id_as_index=doc_id_as_index,
            doc_id_column=doc_id_column,
            flatten_arrays=flatten_arrays,
            flatten_hashmaps=flatten_hashmaps,
        )
