"""SciCode test data helpers (from scicode-bench/SciCode)."""
from __future__ import annotations

import ast
import re
from typing import Any

import h5py
import numpy as np
import scipy


def process_hdf5_list(group: h5py.Group) -> list[Any]:
    data: list[Any] = []
    for key in group.keys():
        data.append(group[key][()])
    return data


def process_hdf5_sparse_matrix(group: h5py.Group) -> scipy.sparse.spmatrix:
    data = group["data"][()]
    shape = tuple(group["shape"][()])
    if "row" in group and "col" in group:
        row = group["row"][()]
        col = group["col"][()]
        return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    if "blocksize" in group:
        indices = group["indices"][()]
        indptr = group["indptr"][()]
        blocksize = tuple(group["blocksize"][()])
        return scipy.sparse.bsr_matrix((data, indices, indptr), shape=shape, blocksize=blocksize)
    indices = group["indices"][()]
    indptr = group["indptr"][()]
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def process_hdf5_dict(group: h5py.Group) -> dict[Any, Any]:
    data: dict[Any, Any] = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            data[key] = process_hdf5_sparse_matrix(obj["sparse_matrix"])
        elif isinstance(obj[()], bytes):
            data[key] = obj[()].decode("utf-8", errors="strict")
        else:
            try:
                float_key = float(key)
                data[float_key] = obj[()]
            except ValueError:
                data[key] = obj[()]
    return data


def process_hdf5_datagroup(group: h5py.Group) -> Any:
    for key in group.keys():
        if key == "list":
            return process_hdf5_list(group[key])
        if key == "sparse_matrix":
            return process_hdf5_sparse_matrix(group[key])
        return process_hdf5_dict(group)
    return {}


def process_hdf5_to_tuple(step_id: str, test_num: int, h5py_file: str) -> list[Any]:
    data_list: list[Any] = []
    with h5py.File(h5py_file, "r") as f:
        for test_id in range(test_num):
            group_path = f"{step_id}/test{test_id + 1}"
            if isinstance(f[group_path], h5py.Group):
                group = f[group_path]
                num_keys = [key for key in group.keys()]
                if len(num_keys) == 1:
                    subgroup = group[num_keys[0]]
                    if isinstance(subgroup, h5py.Dataset):
                        if isinstance(subgroup[()], bytes):
                            data_list.append(subgroup[()].decode("utf-8", errors="strict"))
                        else:
                            data_list.append(subgroup[()])
                    elif isinstance(subgroup, h5py.Group):
                        data_list.append(process_hdf5_datagroup(subgroup))
                else:
                    var_list: list[Any] = []
                    for key in group.keys():
                        subgroup = group[key]
                        if isinstance(subgroup, h5py.Dataset):
                            if isinstance(subgroup[()], bytes):
                                var_list.append(subgroup[()].decode("utf-8", errors="strict"))
                            else:
                                var_list.append(subgroup[()])
                        elif isinstance(subgroup, h5py.Group):
                            var_list.append(process_hdf5_datagroup(subgroup))
                    data_list.append(tuple(var_list))
            else:
                raise FileNotFoundError(f"Path {group_path} not found in HDF5 file.")
    return data_list


def extract_function_name(function_header: str) -> str:
    match = re.search(r"\bdef\s+(\w+)\s*\(", function_header)
    if match:
        return match.group(1)
    match = re.search(r"\bclass\s+(\w+)\s*\(", function_header)
    if match:
        return match.group(1)
    raise ValueError("Function name or class name not found.")


def get_function_from_code(code_string: str | None, function_name: str) -> str | None:
    if code_string is None:
        return None
    try:
        tree = ast.parse(code_string)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == function_name:
                return ast.unparse(node)
    except Exception:
        return code_string
    return code_string
