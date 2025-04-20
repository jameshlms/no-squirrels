try:
    import firebase_admin
    from firebase_admin import firestore
    import google.cloud.firestore_v1 as firestore_v1
    from google.cloud.firestore_v1 import (
        CollectionReference,
        DocumentSnapshot,
        FieldFilter,
        Or,
        And,
        Query,
    )

    FIREBASE_ADMIN_AVAILABLE = True

except ImportError as e:
    firebase_admin = None
    firestore = None
    FIREBASE_ADMIN_AVAILABLE = False

from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Sequence,
    Mapping,
    Set,
    Tuple,
    Union,
    TypeVar,
)
from pandas import DataFrame
import numpy as np

from datetime import datetime
from warnings import warn
from functools import partial

Dtype = Any  # Placeholder for the actual Dtype type from pandas
DOC_ID = Literal["doc_id", "docid", "document_id", "documentid"]
BaseFilter = Union["FieldFilter", "Or", "And"]
T = TypeVar("T")
FirestoreDocValue = Union[
    int,
    float,
    np.int64,
    np.float64,
    str,
    bool,
    list,
    Tuple,
    Set,
    np.ndarray,
    Dict,
    datetime,
]


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
        raise ImportError(
            "The 'firebase_admin' package is not installed. To use firestore features of this package, install 'firebase_admin' using 'pip install firebase-admin'."
        )


def _key_in_doc(key: str, doc: firestore_v1.DocumentSnapshot) -> bool:
    """
    Check if the key is in the document.
    """
    try:
        doc.get(key)
        return True
    except KeyError:
        return False


def _key_in_included_cols(key: str, cols: Set[str]) -> bool:
    if cols is None:
        return True

    if key in cols:
        return True

    return False


def _key_not_in_excluded_cols(key: str, cols: Set[str]) -> bool:
    if cols is None:
        return True

    if key in cols:
        return False

    return True


def _get_incl_cols_record(
    doc: DocumentSnapshot,
    incl_cols: Set[str],
    on_bad_doc: Literal["error", "warn", "skip"] = "error",
) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    for key in incl_cols:
        if not _key_in_doc(key, doc):
            if on_bad_doc == "error":
                raise KeyError(f"Document {doc.id} does not contain the key '{key}'.")
            elif on_bad_doc == "warn":
                warn(
                    f"Document {doc.id} does not contain the key '{key}'. Skipping this document."
                )
                return {}
            else:
                continue

        else:
            record[key] = doc.get(key)

    return record


def _get_excl_cols_record(
    doc: DocumentSnapshot,
    excl_cols: Set[str],
) -> Dict[str, Any]:
    record = {k: v for k, v in doc.to_dict().items() if k not in excl_cols}

    return record


def _deter_key_specification(
    specified_keys: Sequence[str],
) -> Set[str]:
    if isinstance(specified_keys, (list, tuple, set)):
        return set(specified_keys)  # Convert iterable to set
    elif callable(specified_keys):
        raise NotImplementedError(
            "Callable usekeys is not implemented yet. Please provide a list, tuple, set, or dict."
        )
    elif isinstance(specified_keys, dict):
        return set(specified_keys.keys())  # Convert keys to set
    else:
        raise TypeError(
            "The value for usekeys provided was not a valid type. Expected str, list, tuple, set, dict, or callable that accepts a str and returns a bool."
        )


def _flatten_structure(
    structure: Union[Dict, List], key_name: str
) -> Dict[str, FirestoreDocValue]:
    """
    Flatten a structure (dict or list) into a flat dictionary.
    """
    if isinstance(structure, dict):
        return {(key_name + "." + k): v for k, v in structure.items()}
    elif isinstance(structure, list):
        return {(key_name + "." + str(i)): v for i, v in enumerate(structure)}
    else:
        raise TypeError(
            f"Expected a dict or list for {key_name}, but got {type(structure)}."
        )


def _flatten_array(
    value: FirestoreDocValue,
    key_name: str,
) -> Dict[str, FirestoreDocValue]:
    result = {key_name + "." + str(i): v for i, v in enumerate(value)}
    return result


def _flatten_map(
    value: FirestoreDocValue,
    key_name: str,
) -> Dict[str, FirestoreDocValue]:
    result = {key_name + "." + k: v for k, v in value.items()}
    return result


def _flatten_geopoint(
    value: FirestoreDocValue,
    key_name: str,
) -> Dict[str, FirestoreDocValue]:
    return {
        key_name + ".latitude": value.latitude,
        key_name + ".longitude": value.longitude,
    }


def read_firestore(
    app_or_client_or_collection,
    collection: str | Tuple[str, ...],
    index_col: str | Sequence[str | int] | None = None,
    usekeys: (
        Sequence[str]
        | Callable[[str], bool]
        | Mapping[str, str | Callable[[str], str]]
        | None
    ) = None,
    dtypes: Dtype | Mapping[str, Dtype] = None,
    *,
    excludekeys: Sequence[str] | Callable[[str], bool] | None = None,
    filter: BaseFilter | Sequence[BaseFilter] | None = None,
    limit: int | None = None,
    doc_id_as_index: bool = False,
    doc_id_into_col: bool | str = False,
    tags_key_as_cols: str | Sequence[str] | None = None,
    on_bad_doc: Literal["error", "warn", "skip"] = "error",
    flatten_arrays: bool = True,
    flatten_maps: bool = True,
    flatten_geopoints: bool = True,
    flatten_before_doc_processing: bool = True,
) -> DataFrame:
    """Fetch data from Firestore and return it as a DataFrame.

    Args:
        app (firebase_admin.App, firestore_v1.Client): Firebase app instance or Firestore client instance to read data from. If a Firestore client is provided, it will be used directly, otherwise a client will be made with the provided app.
        collection (str, tuple): Name of the collection to read documents from. If a tuple of strings is provided, it will be unpacked as the path to the collection. Argument format required should match Firestore client's `.collection()` method.
        index_col (str, sequence): Format by which to make the dataframe index. If a string is provided, the document key's value will be used as the index. Every document should have this key, otherwise an error will be thrown. If a sequence of integers is provided, the number will sequentially be assigned as a document's index. If None, the default index will be used.
        usekeys (sequence, callable, mapping): Format by which to select the keys to be used in the dataframe. If a sequence of strings is provided, only the keys in the sequence will be used. If a callable is provided, it will be called with the key as an argument and should return True or False. If a mapping is provided, the keys in the mapping will be used and the values will be used as the new keys in the dataframe.
        dtypes (dtype, mapping): Format by which to set the dtypes of the dataframe. If a dtype is provided, it will be used for all columns. If a mapping is provided, the keys in the mapping will be used as the columns and the values will be used as the dtypes.
        filter (firestore_v1.BaseFilter): filter or filters to execute. If provided as a sequence of filters, queries will be executed in the order they are provided. Rules and limitations of filters are inherited from Firestore's client library.
        limit: Maximum number of documents to return.
        doc_id_as_index (bool): If True, the document ID will be used as the index. If False, the document ID will be included as a column.
        include_doc_id (str, bool): If provided, the document ID will be included as a column with a specified name if provided. If True, the document ID will be included as a column with the name '_doc_id'. If False, the document ID will not be included as a column. If a string is provided, the document ID will be included as a column with the specified name.
        tags_as_cols (str, squence): Keys provided will have their array values flattened in a particular ways. If the value associated with the key(s) provided is an array, values in the array will be turned into columns with values being either True or False.
        on_bad_doc (str): If 'error', an error will be raised if a document does not have the key(s) provided. If 'warn', a warning will be raised if a document does not have the key(s) provided but the document stream will continue. If 'skip', the document will be skipped if it does not have the key(s) provided.
        flatten_arrays (bool): If True, arrays will be flattened into separate columns. Set to False by default.
        flatten_maps (bool): If True, maps will be flattened into separate columns. Set to True by default.

    Returns:
        List of documents matching the filter.

    Examples:
    """

    # -----------
    # Base checks
    # -----------

    _check_firebase_admin_available()

    # ------------------------------------------------------------
    # Gets firestore client and retrieves the specified collection
    # ------------------------------------------------------------

    db: firestore_v1.Client = (
        firestore.client(
            app_or_client_or_collection
        )  # If given an app, get the client from it
        if isinstance(app_or_client_or_collection, firebase_admin.App)
        else app_or_client_or_collection  # If given a client, use it directly
    )

    docs: CollectionReference = db.collection(
        *(
            collection  # Unpack the collection if it's a tuple repr of the path
            if isinstance(collection, tuple)
            else (collection,)  # Create a tuple to avoid unpacking string
        )
    )

    # ---------------------
    # Filter the collection
    # ---------------------

    if limit is not None:  # Limit
        docs = docs.limit(limit)

    if filter is not None:  # Filter(s)
        docs = docs.where(filter=filter)

    # -------------------------------------
    # Determining columns for the DataFrame
    # -------------------------------------

    has_specific_cols: bool = False  # Only true if usekeys or excludekeys are provided
    included_cols: Set | None = None
    excluded_cols = set()
    # col_incl_func = lambda col: True
    # not_col_excl_func = lambda col: True

    if usekeys is not None:
        included_cols = _deter_key_specification(usekeys)
        has_specific_cols = True

    if excludekeys is not None:
        excluded_cols = _deter_key_specification(excludekeys)
        has_specific_cols = True

    col_filters: List[Callable[[DocumentSnapshot], bool]] = []
    # to_incl_col_record: List[Callable[[DocumentSnapshot], Dict[str, Any]]] = []

    if has_specific_cols:
        if isinstance(included_cols, set) and any(included_cols):
            included_cols = included_cols - excluded_cols
            col_filters.append(partial(_key_in_included_cols, cols=included_cols))
        elif any(excluded_cols):
            col_filters.append(partial(_key_not_in_excluded_cols, cols=excluded_cols))

    # -------------------------------
    # Preparing indices the DataFrame
    # -------------------------------

    indices = list()
    existing_indices = set()
    records = list()

    # ----------------------------------------------------------------------------
    # Getting the documents from Firestore and transforming them for the DataFrame
    # ----------------------------------------------------------------------------

    doc_stream = docs.stream()

    for doc in doc_stream:
        doc: DocumentSnapshot  # For type hinting and intellisense

        if not doc.exists:
            continue

        if not has_specific_cols:
            record: Dict[str, FirestoreDocValue] = doc.to_dict()
        else:
            if included_cols is not None:
                record = _get_incl_cols_record(
                    doc=doc,
                    incl_cols=included_cols,
                    on_bad_doc=on_bad_doc,
                )
            else:
                record = _get_excl_cols_record(
                    doc=doc,
                    excl_cols=excluded_cols,
                )

        records.append(record)

    return DataFrame.from_records(
        records,
        index=indices if indices else None,
    )


def _safe_get_value(
    doc: DocumentSnapshot | Dict[str, FirestoreDocValue],
    key: str,
    default: FirestoreDocValue = None,
    keep_null_fields: bool = True,
) -> Tuple[FirestoreDocValue, bool]:
    """
    Safely get a value from a document. Returns a tuple of the value and a boolean indicating if the value exists.
    """
    if isinstance(doc, dict):
        value = doc.get(key)
        if not keep_null_fields and value is None:
            value = value or default
        return value, key in doc

    try:
        value = doc.get(key) if isinstance(doc, DocumentSnapshot) else doc[key]
    except KeyError:
        return default, False

    return value if keep_null_fields else (value or default), True


def _make_tag_cols(
    key: str,
    value: FirestoreDocValue,
) -> Dict[str, FirestoreDocValue]:
    """
    Create tag columns from a key and value.
    """
    if isinstance(value, list):
        return {key + "." + v: True for v in value} if value else {}

    elif isinstance(value, dict):
        return {key + "." + k: v for k, v in value.items() if isinstance(v, bool)}

    raise TypeError(f"Expected a list for {key}, but got {type(value)}.")


def _docs_to_records_generator(
    docs: CollectionReference | Query,
    usekeys: Set[str] = set(),
    excludekeys: Set[str] | None = None,
    renamekeys: Dict[str, str] = {},
    tag_like_keys: Set[str] | None = None,
    na_defaults: Dict[str, Any] = {},
    keep_null_fields: bool = True,
    flatten_arrays: bool = True,
    flatten_maps: bool = True,
    flatten_geopoints: bool = True,
    return_doc_id: bool = False,
) -> Generator[Tuple[Dict[str, FirestoreDocValue], str], None, None]:
    """Generator that provides conversions of Firestore documents to records with alterations made as efficient as possible.
    Args:
        docs (CollectionReference | Query): Firestore collection or query to read documents from.
        usekeys (Set[str]): Set of keys to include in the records.
        excludekeys (Set[str] | None): Set of keys to exclude from the records.
        renamekeys (Dict[str, str]): Dictionary of keys to rename in the records.
        tag_like_keys (Set[str] | None): Set of keys to treat as tags.
        na_defaults (Dict[str, Any]): Dictionary of default values for missing keys.
        keep_null_fields (bool): Whether to keep null fields in the records.
        flatten_arrays (bool): Whether to flatten arrays in the records.
        flatten_maps (bool): Whether to flatten maps in the records.
        flatten_geopoints (bool): Whether to flatten geopoints in the records.

    Yields:
        A record with the specified keys and values.
    """
    for doc in docs.stream():

        # -------------------
        # Prepping the record
        # -------------------

        doc_id = doc.id

        doc: DocumentSnapshot | Dict  # For type hinting and intellisense

        record = dict()

        # -------------------------------------------------------------------------------------------
        # Determine if to limit keys and read from the document or convert to dict and get keys after
        # -------------------------------------------------------------------------------------------

        keys = usekeys or (doc := doc.to_dict()).keys()

        # -----------------------------------------
        # Filter keys and values based on arguments
        # -----------------------------------------

        for key in keys:

            if key in excludekeys:
                continue

            value, key_exists = _safe_get_value(
                doc,
                key,
                default=na_defaults.get(key),
                keep_null_fields=keep_null_fields,
            )

            if not key_exists and value is None:
                continue

            key = renamekeys.get(key, key)

            if (
                tag_like_keys
                and key in tag_like_keys
                and isinstance(value, (list, dict))
            ):
                record |= _make_tag_cols(key, value)
                continue

            if flatten_arrays and isinstance(value, list):
                record |= _flatten_array(key, value)
                continue

            if flatten_maps and isinstance(value, dict):
                record |= _flatten_map(key, value)
                continue

            if flatten_geopoints and isinstance(value, dict):
                record |= _flatten_geopoint(key, value)
                continue

            record[key] = value

        # -------------------------------------------
        # Yield the record once altered and assembled
        # -------------------------------------------
        if return_doc_id:
            yield record, doc_id
        else:
            yield record


def new_read_firestore(
    app_or_client_or_collection,
    collection: str | Tuple[str, ...],
    index_col: str | None = None,
    usekeys: (
        Sequence[str]
        | Callable[[str], bool]
        | Mapping[str, str | Callable[[str], str]]
        | None
    ) = None,
    dtypes: Dtype | Mapping[str, Dtype] = None,
    *,
    excludekeys: Sequence[str] | Callable[[str], bool] | None = None,
    filter: BaseFilter | Sequence[BaseFilter] | None = None,
    limit: int | None = None,
    keep_null_fields: bool = True,
    na_defaults: Dict[str, Any] = {},
    renamekeys: Dict[str, str] = {},
    doc_id_as_index: bool = False,
    doc_id_into_col: bool | str = False,
    tags_key_as_cols: str | Sequence[str] | None = None,
    on_bad_doc: Literal["error", "warn", "skip"] = "error",
    flatten_arrays: bool = True,
    flatten_maps: bool = True,
    flatten_geopoints: bool = True,
    processing_order: Tuple[str, str, str] = ("na_defaults", "rename", "flatten"),
):
    # -----------
    # Base checks
    # -----------

    _check_firebase_admin_available()

    # ------------------------------------------------------------
    # Gets firestore client and retrieves the specified collection
    # ------------------------------------------------------------

    db: firestore_v1.Client = (
        firestore.client(
            app_or_client_or_collection
        )  # If given an app, get the client from it
        if isinstance(app_or_client_or_collection, firebase_admin.App)
        else app_or_client_or_collection  # If given a client, use it directly
    )

    docs: CollectionReference = db.collection(
        *(
            collection  # Unpack the collection if it's a tuple repr of the path
            if isinstance(collection, tuple)
            else (collection,)  # Create a tuple to avoid unpacking string
        )
    )

    if not doc_id_as_index:
        df = DataFrame(_docs_to_records_generator())

        if not index_col:
            return df

        if df[index_col].isna().any():
            raise KeyError(f"Document {doc_id} does not contain the key '{index_col}'.")

        if df[index_col].duplicated().any():
            raise KeyError(f"Two documents contain the same value at '{index_col}.")

        return df.set_index(index_col, inplace=True)

    df = DataFrame()

    docs_to_records_kwargs = {
        "usekeys": usekeys,
        "excludekeys": excludekeys,
        "renamekeys": {},
        "tag_like_keys": tags_key_as_cols,
        "na_defaults": na_defaults,
        "keep_null_fields": keep_null_fields,
        "flatten_arrays": flatten_arrays,
        "flatten_maps": flatten_maps,
        "flatten_geopoints": flatten_geopoints,
        "return_doc_id": True,
    }

    existing_indices = set()

    for record, doc_id in _docs_to_records_generator(docs, **docs_to_records_kwargs):

        if doc_id_into_col:
            if isinstance(doc_id_into_col, str):
                record[doc_id_into_col] = doc_id

            else:
                record["_DOC_ID"] = doc_id

        if doc_id_as_index:
            df.loc[doc_id] = record
            continue

        if not index_col:
            continue

        if not index_col in record:
            raise KeyError(f"Document {doc_id} does not contain the key '{index_col}'.")
        if index_col in existing_indices:
            raise KeyError(
                f"Document {doc_id} contains the key '{index_col}' with a duplicate value '{record[index_col]}'."
            )
        df.loc[record.pop(index_col)] = record


class FirestoreReader:

    def __init__(self, app_or_client):
        _check_firebase_admin_available()
        _check_arg_type(
            app_or_client, "app_or_client", (firebase_admin.App, firestore.Client)
        )
        self.db = (
            firestore.client(app_or_client)
            if isinstance(app_or_client, firebase_admin.App)
            else app_or_client
        )

    def set_client(self, app_or_client) -> "FirestoreReader":
        _check_firebase_admin_available()
        _check_arg_type(
            app_or_client, "app_or_client", (firebase_admin.App, firestore.Client)
        )
        self.db = (
            firestore.client(app_or_client)
            if isinstance(app_or_client, firebase_admin.App)
            else app_or_client
        )
        return self

    def set_collection(self, collection: str) -> "FirestoreReader":
        _check_arg_type(collection, "collection", str)
        self.collection = self.db.collection(collection)
        return self

    def get_frame(
        self,
        *,
        collection: str | None = None,
        filter: FieldFilter = None,
        limit: int = None,
        key_as_index: str | None = None,
        doc_id_as_index: bool = False,
        doc_id_as_col: str | None = None,
        flatten_arrays: bool = False,
        flatten_maps: bool = True,
    ) -> DataFrame:
        _check_arg_type(collection, "collection", str, allow_none=True)
        if collection is not None:
            self.set_collection(collection)

        return read_firestore(
            app_or_client=self.db,
            collection=self.collection.id,
            filter=filter,
            limit=limit,
            key_as_index=key_as_index,
            doc_id_as_index=doc_id_as_index,
            doc_id_as_col=doc_id_as_col,
            flatten_arrays=flatten_arrays,
            flatten_maps=flatten_maps,
        )
